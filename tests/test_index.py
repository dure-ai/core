"""The indexer: records embedded in append order, grouped per (scope, owner),
and read back through a SimilarityReader that respects visibility."""

from datetime import UTC, datetime

import pytest
from spy import SpyBackend, SpyEmbedder, TxnSpy

from cortex_mem.contract import Caller, Event, Group, Knowledge
from cortex_mem.embed import EmbeddingError, HashEmbedder, cosine, float32, text_digest
from cortex_mem.index import WARM_BATCH, Index, IndexConfigChanged
from cortex_mem.store.base import CursorConflict
from cortex_mem.store.memory import InMemoryDatabase
from cortex_mem.store.sqlite import SqliteDatabase

T0 = datetime(2026, 11, 15, tzinfo=UTC)


def ev(id, text="deploy uses tags", kind="record", user="u1", team="t1", scope="personal", **kw):
    return Event(id=id, kind=kind, at=T0, user=user, team=team, session="s1", scope=scope, text=text, **kw)


def caller(user="u1", team="t1"):
    return Caller(user=user, team=team, session="s2", at=T0)


def item(id, scope="team", owner="t1", title="deploy", body="deploy uses tags"):
    return Knowledge(id=id, kind="knowledge", title=title, body=body, scope=scope, owner=owner, sources=("e1",))


class CountingEmbedder:
    """HashEmbedder that remembers every text it was asked to embed."""

    def __init__(self, inner=None):
        self.inner = inner or HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.calls = []  # (text, kind)

    def embed(self, texts, kind):
        self.calls.extend((t, kind) for t in texts)
        return self.inner.embed(texts, kind)


class FailingEmbedder(CountingEmbedder):
    """Raises EmbeddingError when asked to embed one of the given texts."""

    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = set(fail_on)

    def embed(self, texts, kind):
        if any(t in self.fail_on for t in texts):
            raise EmbeddingError("embedding backend is down")
        return super().embed(texts, kind)


class TableEmbedder:
    """Text -> vector from a table, so a test controls every score exactly."""

    model = "table"
    dim = 3

    def __init__(self, table):
        self.table = {text: [float(x) for x in vec] for text, vec in table.items()}
        self.calls = []

    def embed(self, texts, kind):
        self.calls.extend((t, kind) for t in texts)
        return [list(self.table[t]) for t in texts]


def build(events, embedder=None, threshold=0.9, db=None):
    db = db or InMemoryDatabase()
    for e in events:
        db.events.append(e)
    index = Index(db.vectors, embedder or HashEmbedder(), threshold, transaction=db.transaction)
    index.ensure_current(db.knowledge)
    index.advance(db.events, len(events))
    return db, index


def put_knowledge(db, index, *items):
    """Put items in the knowledge base and let the index follow them."""
    for k in items:
        db.knowledge.put(k)
    index.sync_knowledge(db.knowledge, [k.id for k in items])


def reader(db, index):
    return index.reader(db.events, db.knowledge)


def ids(pairs):
    return [x.id for x, _ in pairs]


# ── grouping ────────────────────────────────────────────────────────────────


def test_first_record_starts_its_own_group():
    db, index = build([ev("e1", scope="team")])
    assert reader(db, index).group("e1") == Group("e1", "team", "t1", ("e1",))


def test_close_record_joins_the_nearest_members_group():
    emb = TableEmbedder({"a": [1, 0, 0], "b": [0.95, (1 - 0.95**2) ** 0.5, 0]})
    assert cosine(emb.table["a"], emb.table["b"]) >= 0.9
    db, index = build([ev("e1", "a"), ev("e2", "b")], emb, threshold=0.9)
    r = reader(db, index)
    assert r.group("e2") == Group("e1", "personal", "u1", ("e1", "e2"))
    assert r.group("e1") == r.group("e2")


def test_record_below_the_threshold_starts_a_new_group():
    emb = TableEmbedder({"a": [1, 0, 0], "b": [0.85, (1 - 0.85**2) ** 0.5, 0]})
    db, index = build([ev("e1", "a"), ev("e2", "b")], emb, threshold=0.9)
    r = reader(db, index)
    assert r.group("e2").id == "e2"
    assert r.group("e2").members == ("e2",)
    assert r.group("e1").members == ("e1",)


def test_score_equal_to_the_threshold_joins():
    emb = TableEmbedder({"a": [1, 0, 0], "b": [0.6, 0.8, 0]})
    threshold = cosine(float32(emb.table["b"]), float32(emb.table["a"]))  # the score the store computes
    db, index = build([ev("e1", "a"), ev("e2", "b")], emb, threshold=threshold)
    assert reader(db, index).group("e2").members == ("e1", "e2")


def test_tie_goes_to_the_earlier_member():
    emb = TableEmbedder({"a": [1, 0, 0], "b": [0, 1, 0], "c": [1, 1, 0]})
    assert cosine(emb.table["c"], emb.table["a"]) == cosine(emb.table["c"], emb.table["b"])
    db, index = build([ev("e1", "a"), ev("e2", "b"), ev("e3", "c")], emb, threshold=0.7)
    r = reader(db, index)
    assert r.group("e2").id == "e2"
    assert r.group("e3") == Group("e1", "personal", "u1", ("e1", "e3"))


def test_groups_never_cross_scope_or_owner():
    events = [
        ev("p1", user="u1", team="t1", scope="personal"),
        ev("p2", user="u2", team="t1", scope="personal"),
        ev("t1", user="u1", team="t1", scope="team"),
        ev("t2", user="u3", team="t2", scope="team"),
        ev("o1", user="u3", team="t2", scope="org"),
        ev("o2", user="u1", team="t1", scope="org"),
    ]
    db, index = build(events)
    r = reader(db, index)
    assert r.group("p1") == Group("p1", "personal", "u1", ("p1",))
    assert r.group("p2") == Group("p2", "personal", "u2", ("p2",))
    assert r.group("t1") == Group("t1", "team", "t1", ("t1",))
    assert r.group("t2") == Group("t2", "team", "t2", ("t2",))
    # org has one owner, so the same text from different teams does group.
    assert r.group("o2") == Group("o1", "org", "*", ("o1", "o2"))


def test_conditions_do_not_split_groups():
    events = [
        ev("e1", conditions={"service": "api"}),
        ev("e2", conditions={"service": "web"}),
    ]
    db, index = build(events)
    assert reader(db, index).group("e2") == Group("e1", "personal", "u1", ("e1", "e2"))


def test_events_that_are_not_embedded_get_no_vector_or_group():
    events = [
        ev("e1"),
        ev("r1", kind="recall"),
        ev("o1", kind="outcome", outcome="success", signal="test", refs=("e1",)),
        ev("h1", kind="human_confirm", refs=("e1",)),
        ev("c1", kind="commit"),
        ev("x1", text_redacted=True),
        ev("x2", text=""),
    ]
    emb = CountingEmbedder()
    db, index = build(events, emb)
    for id in ["r1", "o1", "h1", "c1", "x1", "x2"]:
        assert db.vectors.vector("event", id) is None
        assert db.vectors.group_of(id) is None
        assert reader(db, index).group(id) is None
    assert db.vectors.indexed() == len(events)
    assert [id for id, _ in db.vectors.vectors("event")] == ["e1"]
    assert len(emb.calls) == 1


def test_cache_avoids_a_second_embedding_of_the_same_text():
    emb = CountingEmbedder()
    db, _ = build([ev("e1", "same words"), ev("e2", "same words", user="u2")], emb)
    assert emb.calls == [("same words", "passage")]
    assert db.vectors.vector("event", "e2") == db.vectors.vector("event", "e1")
    assert db.vectors.cached(emb.model, text_digest("same words")) == db.vectors.vector("event", "e1")


def test_index_gives_the_same_groups_on_sqlite():
    events = [
        ev("e1", "deploy uses tags"),
        ev("e2", "deploy uses tags", conditions={"service": "api"}),
        ev("e3", "payments need vendor v2"),
        ev("e4", "deploy uses tags", scope="team"),
    ]
    mem_db, mem_index = build(events, threshold=0.9)
    sql_db, sql_index = build(events, threshold=0.9, db=SqliteDatabase())
    for e in events:
        assert reader(sql_db, sql_index).group(e.id) == reader(mem_db, mem_index).group(e.id)
    assert list(sql_db.vectors.vectors("event")) == list(mem_db.vectors.vectors("event"))


# ── knowledge and index version ─────────────────────────────────────────────


def test_knowledge_vectors_follow_the_knowledge_base():
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9)
    emb = HashEmbedder()
    put_knowledge(db, index, item("k1", title="deploy", body="uses tags"))
    assert db.vectors.vector("knowledge", "k1") == float32(emb.embed(["deploy\nuses tags"], "passage")[0])
    assert db.vectors.knowledge_digests() == {"k1": text_digest("deploy\nuses tags")}
    put_knowledge(db, index, item("k1", title="deploy", body="uses branches"))
    assert db.vectors.vector("knowledge", "k1") == float32(emb.embed(["deploy\nuses branches"], "passage")[0])
    db.knowledge.remove("k1")
    index.sync_knowledge(db.knowledge, ["k1", "never-there"])
    assert db.vectors.vector("knowledge", "k1") is None
    assert list(db.vectors.vectors("knowledge")) == []


def test_ensure_current_resets_and_reembeds_knowledge():
    emb = CountingEmbedder()
    db, index = build([ev("e1"), ev("e2", "other words")], emb)  # built and advanced
    put_knowledge(db, index, item("k1"), item("k2", scope="personal", owner="u9", body="private note"))
    db.vectors.put_vector("knowledge", "gone", [1.0, 0.0])  # not in the knowledge base
    generation = db.vectors.generation()
    calls = len(emb.calls)
    assert index.ensure_current(db.knowledge) is False  # current: nothing to do
    assert db.vectors.indexed() == 2

    other = Index(db.vectors, emb, 0.8)
    assert other.ensure_current(db.knowledge) is True
    assert db.vectors.meta() == other.want() == {"model": emb.model, "group_threshold": 0.8, "store": "memory"}
    assert db.vectors.generation() == generation + 1
    assert list(db.vectors.vectors("event")) == []
    assert db.vectors.group_of("e1") is None
    assert db.vectors.indexed() == 0
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1", "k2"]
    assert set(db.vectors.knowledge_digests()) == {"k1", "k2"}
    assert len(emb.calls) == calls  # same model: every passage comes from the cache
    other.advance(db.events, 2)
    assert len(emb.calls) == calls
    assert reader(db, other).group("e1").id == "e1"


def test_stale_when_model_or_threshold_changes():
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9)
    assert index.stale()  # never built
    assert index.ensure_current(db.knowledge) is True
    assert not index.stale()
    assert Index(db.vectors, HashEmbedder(), 0.9).stale() is False
    assert Index(db.vectors, HashEmbedder(), 0.8).stale()
    assert Index(db.vectors, HashEmbedder(dim=128), 0.9).stale()


# ── SimilarityReader ────────────────────────────────────────────────────────


def visibility_world():
    events = [
        ev("mine", user="u1", team="t1", scope="personal"),
        ev("theirs", user="u2", team="t1", scope="personal"),
        ev("my-team", user="u2", team="t1", scope="team"),
        ev("other-team", user="u3", team="t2", scope="team"),
        ev("org", user="u3", team="t2", scope="org"),
        ev("my-team-by-me", user="u1", team="t1", scope="team"),
    ]
    return build(events)


def test_search_respects_visibility():
    db, index = visibility_world()
    r = reader(db, index)
    me = caller("u1", "t1")
    assert ids(r.search("deploy uses tags", me, "personal")) == ["mine", "my-team-by-me"]
    assert ids(r.search("deploy uses tags", me, "team")) == ["mine", "my-team", "my-team-by-me"]
    assert ids(r.search("deploy uses tags", me, "org")) == ["mine", "my-team", "org", "my-team-by-me"]
    for scope in ["personal", "team", "org"]:
        found = ids(r.search("deploy uses tags", me, scope))
        assert "theirs" not in found
        assert "other-team" not in found
    assert ids(r.search("deploy uses tags", caller("u2", "t1"), "personal")) == ["theirs", "my-team"]


def test_search_filters_kinds():
    db, index = visibility_world()
    r = reader(db, index)
    me = caller()
    assert ids(r.search("deploy uses tags", me, "team", kinds=["record"])) == ["mine", "my-team", "my-team-by-me"]
    assert r.search("deploy uses tags", me, "team", kinds=["recall", "outcome"]) == []


def test_search_ranks_by_score_drops_non_positive_and_cuts_to_k():
    emb = TableEmbedder({
        "q": [1, 0, 0],
        "zero": [0, 1, 0],
        "half": [1, 1, 0],
        "exact": [1, 0, 0],
        "opposite": [-1, 0, 0],
        "close": [1, 0.2, 0],
        "nothing": [0, 0, 0],
    })
    texts = ["zero", "half", "exact", "opposite", "close", "nothing"]
    db, index = build([ev(t, t) for t in texts], emb, threshold=2.0)  # no grouping
    r = reader(db, index)
    found = r.search("q", caller(), "personal")
    assert ids(found) == ["exact", "close", "half"]
    assert [s for _, s in found] == pytest.approx([1.0, 1 / (1.04**0.5), 1 / 2**0.5])
    assert ids(r.search("q", caller(), "personal", k=2)) == ["exact", "close"]
    assert r.search("q", caller(), "personal", k=0) == []
    assert ("q", "query") in emb.calls


def test_search_ties_keep_append_order():
    emb = TableEmbedder({"q": [1, 0, 0], "a": [1, 0, 0], "b": [2, 0, 0], "c": [1, 1, 0], "d": [3, 0, 0]})
    db, index = build([ev("e1", "c"), ev("e2", "a"), ev("e3", "b"), ev("e4", "d")], emb, threshold=2.0)
    assert ids(reader(db, index).search("q", caller(), "personal")) == ["e2", "e3", "e4", "e1"]


def test_neighbors_exclude_self_and_what_the_caller_cannot_see():
    db, index = visibility_world()
    r = reader(db, index)
    me = caller("u1", "t1")
    assert ids(r.neighbors("mine", me, "personal")) == ["my-team-by-me"]
    assert ids(r.neighbors("mine", me, "org")) == ["my-team", "org", "my-team-by-me"]
    assert ids(r.neighbors("mine", me, "org", k=1)) == ["my-team"]
    for scope in ["personal", "team", "org"]:
        found = ids(r.neighbors("mine", me, scope))
        assert "mine" not in found
        assert "theirs" not in found
        assert "other-team" not in found


def test_neighbors_of_an_event_without_a_vector_is_empty():
    db, index = build([ev("e1"), ev("r1", kind="recall"), ev("x1", text_redacted=True)])
    r = reader(db, index)
    assert r.neighbors("r1", caller(), "org") == []
    assert r.neighbors("x1", caller(), "org") == []
    assert r.neighbors("unknown", caller(), "org") == []


def test_knowledge_respects_visibility():
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9)
    items = [
        item("a-mine", scope="personal", owner="u1"),
        item("b-theirs", scope="personal", owner="u2"),
        item("c-my-team", scope="team", owner="t1"),
        item("d-other-team", scope="team", owner="t2"),
        item("e-org", scope="org", owner="*"),
    ]
    put_knowledge(db, index, *items)
    r = reader(db, index)
    me = caller("u1", "t1")
    q = "deploy\ndeploy uses tags"
    assert ids(r.knowledge(q, me, "personal")) == ["a-mine"]
    assert ids(r.knowledge(q, me, "team")) == ["a-mine", "c-my-team"]
    assert ids(r.knowledge(q, me, "org")) == ["a-mine", "c-my-team", "e-org"]
    assert ids(r.knowledge(q, me, "org", k=2)) == ["a-mine", "c-my-team"]
    assert all(s == pytest.approx(1.0) for _, s in r.knowledge(q, me, "org"))


def test_knowledge_drops_unrelated_items():
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9)
    put_knowledge(db, index, item("k1", title="deploy", body="uses tags"), item("k2", title="payments", body="vendor v2"))
    assert ids(reader(db, index).knowledge("vendor v2 payments", caller(), "team")) == ["k2"]


def test_query_is_not_stored():
    emb = CountingEmbedder()
    db, index = build([ev("e1")], emb)
    put_knowledge(db, index, item("k1"))
    r = reader(db, index)
    before_vectors = {kind: list(db.vectors.vectors(kind)) for kind in ["event", "knowledge"]}
    r.search("secret plan", caller(), "org")
    r.knowledge("secret plan", caller(), "org")
    digest = text_digest("secret plan")
    assert db.vectors.cached(emb.model, digest) is None
    assert {kind: list(db.vectors.vectors(kind)) for kind in ["event", "knowledge"]} == before_vectors
    for kind in ["event", "knowledge"]:
        assert all(id != digest for id, _ in db.vectors.vectors(kind))
    assert ("secret plan", "query") in emb.calls
    # Embedded once, then served from the in-process LRU.
    assert emb.calls.count(("secret plan", "query")) == 1


def test_query_lru_is_bounded():
    emb = CountingEmbedder()
    db, index = build([ev("e1")], emb)
    r = reader(db, index)
    for i in range(300):
        r.search(f"query {i}", caller(), "org")
    r.search("query 299", caller(), "org")
    assert emb.calls.count(("query 299", "query")) == 1
    r.search("query 0", caller(), "org")  # evicted long ago
    assert emb.calls.count(("query 0", "query")) == 2


def test_empty_query_returns_nothing():
    emb = CountingEmbedder()
    db, index = build([ev("e1")], emb)
    put_knowledge(db, index, item("k1"))
    calls = len(emb.calls)
    r = reader(db, index)
    for q in ["", "   ", "\n"]:
        assert r.search(q, caller(), "org") == []
        assert r.knowledge(q, caller(), "org") == []
    assert len(emb.calls) == calls


def test_reader_is_read_only_and_names_the_model():
    db, index = build([ev("e1")])
    r = reader(db, index)
    assert r.model == "hash-256"
    public = {name for name in dir(r) if not name.startswith("_")}
    assert public == {"model", "search", "neighbors", "knowledge", "group"}


def test_group_of_unknown_event_is_none():
    db, index = build([ev("e1")])
    assert reader(db, index).group("nope") is None


# ── fix round 1 ─────────────────────────────────────────────────────────────


def test_neighbors_of_an_event_the_caller_cannot_see_is_empty():
    db, index = visibility_world()
    r = reader(db, index)
    me = caller("u1", "t1")
    for scope in ["personal", "team", "org"]:
        assert r.neighbors("theirs", me, scope) == []  # u2's personal record
        assert r.neighbors("other-team", me, scope) == []
    assert r.neighbors("my-team", me, "personal") == []  # team record, personal scope
    assert ids(r.neighbors("my-team", me, "team")) == ["mine", "my-team-by-me"]
    assert ids(r.neighbors("theirs", caller("u2", "t1"), "personal")) == ["my-team"]


class CountMismatchEmbedder:
    """Returns a fixed list of vectors whatever it is asked to embed."""

    model = "broken"
    dim = 3

    def __init__(self, out):
        self.out = out

    def embed(self, texts, kind):
        return [list(v) for v in self.out]


@pytest.mark.parametrize("out", [[], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
def test_wrong_vector_count_raises_embedding_error(out):
    db = InMemoryDatabase()
    db.events.append(ev("e1"))
    index = Index(db.vectors, CountMismatchEmbedder(out), 0.9)
    index.ensure_current(db.knowledge)
    with pytest.raises(EmbeddingError):
        index.advance(db.events, 1)
    assert db.vectors.indexed() == 0
    assert db.vectors.vector("event", "e1") is None
    assert db.vectors.cached("broken", text_digest("deploy uses tags")) is None
    db.knowledge.put(item("k1"))
    with pytest.raises(EmbeddingError):
        index.sync_knowledge(db.knowledge, ["k1"])
    assert db.vectors.vector("knowledge", "k1") is None
    r = reader(db, index)
    with pytest.raises(EmbeddingError):
        r.search("deploy", caller(), "org")
    with pytest.raises(EmbeddingError):
        r.knowledge("deploy", caller(), "org")


def test_record_joins_the_closest_member_not_the_first_above_the_threshold():
    emb = TableEmbedder({"a": [1, 0, 0], "b": [0, 1, 0], "c": [0.6, 0.8, 0]})
    db, index = build([ev("e1", "a"), ev("e2", "b"), ev("e3", "c")], emb, threshold=0.5)
    r = reader(db, index)
    assert cosine(emb.table["c"], emb.table["a"]) >= 0.5  # e1 clears the threshold too
    assert r.group("e1").members == ("e1",)
    assert r.group("e3") == Group("e2", "personal", "u1", ("e2", "e3"))


def test_query_lru_keeps_recently_used_queries():
    emb = CountingEmbedder()
    db, index = build([ev("e1")], emb)
    r = reader(db, index)
    for i in range(256):  # fill the cache
        r.search(f"query {i}", caller(), "org")
    r.search("query 0", caller(), "org")  # touch the oldest
    for i in range(256, 356):  # 100 more: FIFO would evict query 0..99
        r.search(f"query {i}", caller(), "org")
    r.search("query 0", caller(), "org")
    assert emb.calls.count(("query 0", "query")) == 1  # still cached
    r.search("query 1", caller(), "org")
    assert emb.calls.count(("query 1", "query")) == 2  # untouched, evicted


def test_negative_k_returns_nothing():
    db, index = visibility_world()
    put_knowledge(db, index, item("k1"))
    r = reader(db, index)
    me = caller()
    assert r.search("deploy uses tags", me, "org", k=-1) == []
    assert r.neighbors("mine", me, "org", k=-1) == []
    assert r.knowledge("deploy\ndeploy uses tags", me, "org", k=-1) == []


# ── a reader bounded to what the engine has seen ────────────────────────────


def test_a_bounded_reader_sees_no_event_after_its_position():
    events = [ev("e1"), ev("o1", kind="outcome", refs=("e1",)), ev("e2"), ev("e3"), ev("x1", scope="team")]
    db, index = build(events)
    full = reader(db, index)
    assert full.group("e3").members == ("e1", "e2", "e3")

    upto3 = index.reader(db.events, db.knowledge, upto=3)  # e1, o1, e2
    assert upto3.group("e1") == Group("e1", "personal", "u1", ("e1", "e2"))
    assert upto3.group("e2") == Group("e1", "personal", "u1", ("e1", "e2"))
    assert upto3.group("e3") is None  # not seen yet
    assert upto3.group("x1") is None
    assert ids(upto3.search("deploy uses tags", caller(), "org")) == ["e1", "e2"]
    assert ids(upto3.neighbors("e1", caller(), "org")) == ["e2"]
    assert upto3.neighbors("e3", caller(), "org") == []

    upto1 = index.reader(db.events, db.knowledge, upto=1)
    assert upto1.group("e1").members == ("e1",)
    assert upto1.neighbors("e1", caller(), "org") == []
    assert ids(index.reader(db.events, db.knowledge, upto=5).search("deploy uses tags", caller(), "org")) == [
        "e1", "e2", "e3", "x1",
    ]


def test_a_bounded_reader_still_searches_all_current_knowledge():
    db, index = build([ev("e1")])
    put_knowledge(db, index, item("k1"))
    assert ids(index.reader(db.events, db.knowledge, upto=0).knowledge("deploy\ndeploy uses tags", caller(), "team")) == [
        "k1"
    ]


# ── the store computes similarity ───────────────────────────────────────────


@pytest.mark.parametrize("make", [InMemoryDatabase, SqliteDatabase], ids=["memory", "sqlite"])
def test_the_index_version_names_the_store(make):
    db = make()
    index = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
    assert index.want() == {"model": "hash-256", "group_threshold": 0.9, "store": db.vectors.kind}
    assert index.want()["store"] == ("memory" if make is InMemoryDatabase else "sqlite")


class NoScanBackend:
    """A VectorBackend that refuses the full scans, so an Index that still
    compares every vector in Python fails here."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name in ("vectors", "partition"):
            raise AssertionError(f"the index scanned with {name}()")
        return getattr(self._inner, name)


def test_the_index_asks_the_store_for_similarity_instead_of_scanning():
    db = InMemoryDatabase()
    events = [ev("e1"), ev("e2"), ev("e3", "payments need vendor v2"), ev("x1", user="u9", team="t9")]
    for e in events:
        db.events.append(e)
    index = Index(NoScanBackend(db.vectors), HashEmbedder(), 0.9)
    index.ensure_current(db.knowledge)
    index.advance(db.events, len(events))
    put_knowledge(db, index, item("k1"))
    r = reader(db, index)

    assert r.group("e2") == Group("e1", "personal", "u1", ("e1", "e2"))
    assert ids(r.search("deploy uses tags", caller(), "org")) == ["e1", "e2"]
    assert ids(r.neighbors("e1", caller(), "org")) == ["e2"]
    assert ids(r.knowledge("deploy\ndeploy uses tags", caller(), "team")) == ["k1"]


@pytest.mark.parametrize("make", [InMemoryDatabase, SqliteDatabase], ids=["memory", "sqlite"])
def test_a_non_finite_embedding_fails_loudly_and_writes_nothing(make):
    db = make()
    db.events.append(ev("e1", "a"))
    emb = TableEmbedder({"a": [float("nan"), 1.0, 0.0]})
    index = Index(db.vectors, emb, 0.9, transaction=db.transaction)
    index.ensure_current(db.knowledge)
    with pytest.raises(ValueError, match="finite"):
        index.advance(db.events, 1)
    assert (list(db.vectors.vectors("event")), db.vectors.group_of("e1"), db.vectors.indexed()) == ([], None, 0)
    assert db.vectors.cached("table", text_digest("a")) is None


# ── indexing outside the write transaction (advance, sync_knowledge) ────────


@pytest.fixture(params=["memory", "sqlite", "pg"])
def opened(request, tmp_path):
    """opened(): a connection to this test's one store. Memory gives the same
    object each time; SQLite and PostgreSQL open a new connection to the same
    file or schema, as a second process would."""
    kind = request.param
    url = request.getfixturevalue("pg_schema") if kind == "pg" else None
    made = []

    def open_db():
        if kind == "memory":
            db = made[0] if made else InMemoryDatabase()
        elif kind == "sqlite":
            db = SqliteDatabase(tmp_path / "core.db")
        else:
            from cortex_mem.store.postgres import PostgresDatabase

            db = PostgresDatabase(url)
        made.append(db)
        return db

    yield open_db
    for db in made:
        if kind == "sqlite":
            db.con.close()
        elif kind == "pg":
            db.close()


def spied(db, embedder=None, threshold=0.9, watch=("nearest_member", "group_of"), built=True):
    """An Index whose transactions, embedder and backend calls are watched."""
    spy = TxnSpy(db)
    emb = embedder or SpyEmbedder(spy)
    emb.spy = spy
    backend = SpyBackend(db.vectors, spy, watch)
    index = Index(backend, emb, threshold, transaction=spy.transaction)
    if built and index.stale():
        db.vectors.reset(index.want())  # an empty index of this config, without embedding anything
    return index, emb, backend, spy


def appended(db, *events):
    for e in events:
        db.events.append(e)
    return db


def test_advance_embeds_outside_any_transaction(opened):
    db = appended(opened(), ev("e1", "one"), ev("e2", "two"), ev("o1", kind="outcome", refs=("e1",)), ev("e3", "one"))
    index, emb, _, _ = spied(db)

    index.advance(db.events, 4)

    assert emb.texts == ["one", "two"]
    assert emb.inside == []
    assert db.vectors.indexed() == 4
    assert reader(db, index).group("e3").members == ("e1", "e3")


def test_advance_indexes_up_to_upto_and_resumes(opened):
    db = appended(opened(), *[ev(f"e{i}", text=f"fact number {i}") for i in range(1, 6)])
    index, emb, backend, _ = spied(db)

    index.advance(db.events, 2)
    assert db.vectors.indexed() == 2
    assert emb.texts == ["fact number 1", "fact number 2"]
    index.advance(db.events, 4)
    assert db.vectors.indexed() == 4
    assert emb.texts[2:] == ["fact number 3", "fact number 4"]
    assert all(kind == "passage" for _, kind, _ in emb.calls)
    assert db.vectors.vector("event", "e5") is None
    assert [e for e in backend.log if e[0] == "put_group"] == [("put_group", f"e{i}") for i in range(1, 5)]


def test_advance_with_upto_at_or_below_indexed_does_nothing(opened):
    db = appended(opened(), ev("e1", "one"), ev("e2", "two"), ev("e3", "three"))
    index, emb, backend, _ = spied(db)
    index.advance(db.events, 3)
    calls, log = len(emb.calls), len(backend.log)

    index.advance(db.events, 3)
    index.advance(db.events, 2)
    index.advance(db.events, 0)

    assert (len(emb.calls), len(backend.log), db.vectors.indexed()) == (calls, log, 3)


def test_advance_failure_leaves_nothing_for_that_event(opened):
    db = appended(opened(), ev("e1", "fine"), ev("e2", "boom"), ev("e3", "also fine"))
    index, emb, backend, _ = spied(db)
    emb.fail_on.add("boom")

    with pytest.raises(EmbeddingError):
        index.advance(db.events, 3)

    # The batch failed, so its texts were tried one at a time: the event
    # before the failing text is indexed, the failing one and after are not.
    assert emb.batches == [["fine", "boom", "also fine"], ["fine"], ["boom"]]
    assert db.vectors.indexed() == 1
    assert [id for id, _ in db.vectors.vectors("event")] == ["e1"]
    for id in ("e2", "e3"):
        assert db.vectors.vector("event", id) is None
        assert db.vectors.group_of(id) is None
    assert db.vectors.cached(emb.model, text_digest("boom")) is None
    assert [e[-1] for e in backend.log] == ["e1", "e1"]
    assert emb.inside == []
    # The next try picks up where it stopped once the backend recovers.
    emb.fail_on.clear()
    index.advance(db.events, 3)
    assert db.vectors.indexed() == 3
    assert [id for id, _ in db.vectors.vectors("event")] == ["e1", "e2", "e3"]


def test_advance_failure_on_the_first_text_indexes_nothing(opened):
    db = appended(opened(), ev("o1", kind="outcome", refs=("x",)), ev("e2", "boom"), ev("e3", "fine"))
    index, emb, backend, _ = spied(db)
    emb.fail_on.add("boom")

    with pytest.raises(EmbeddingError):
        index.advance(db.events, 3)

    assert db.vectors.indexed() == 1  # the outcome before it needs no embedding
    assert backend.log == []


def test_advance_batches_embedder_calls(opened):
    texts = [f"text {i}" for i in range(130)]
    db = appended(
        opened(),
        *[ev(f"e{i}", t) for i, t in enumerate(texts)],
        ev("again", "text 0"),  # cached by the first batch
        ev("o1", kind="outcome", refs=("e0",)),
    )
    index, emb, _, _ = spied(db)

    index.advance(db.events, 132)

    assert [len(b) for b in emb.batches] == [WARM_BATCH, WARM_BATCH, 2]  # missing, distinct, in order
    assert emb.batches[0][0] == "text 0"
    assert emb.inside == []
    assert db.vectors.indexed() == 132
    assert reader(db, index).group("again").members == ("e0", "again")


def test_two_indexers_one_wins(opened):
    """The first indexer embeds; meanwhile a second one (another process)
    indexes part of the same stretch. The first then finds the position moved
    when it writes, reads it again and finishes from there: each event is
    written once."""
    db_a, db_b = opened(), opened()
    appended(db_a, ev("e1", "one"), ev("e2", "two"), ev("o1", kind="outcome", refs=("e1",)), ev("e4", "one"))
    log = []
    second, _, b_backend, _ = spied(db_b)

    def meanwhile(texts, kind):
        if not log:
            log.append("b ran")
            second.advance(db_b.events, 2)

    first, emb_a, a_backend, _ = spied(db_a, SpyEmbedder(None, during=meanwhile))

    first.advance(db_a.events, 4)

    assert log == ["b ran"]
    assert db_a.vectors.indexed() == 4
    writes = [e for e in b_backend.log + a_backend.log if e[0] in ("put_vector", "put_group")]
    assert sorted(writes) == sorted(
        [("put_vector", "event", id) for id in ("e1", "e2", "e4")] + [("put_group", id) for id in ("e1", "e2", "e4")]
    )
    assert [e[-1] for e in b_backend.log] == ["e1", "e1", "e2", "e2"]  # b indexed 1..2
    assert [e[-1] for e in a_backend.log] == ["e4", "e4"]  # a went on from 2, from its cache
    assert emb_a.inside == []
    assert reader(db_a, first).group("e4").members == ("e1", "e4")


def test_advance_raises_a_conflict_it_cannot_explain(opened):
    """A backend whose position never moves (it ignores the shared cursor)
    gets the conflict raised, not an endless retry, and writes nothing."""
    db = appended(opened(), ev("e1", "one"), ev("e2", "two"))
    index = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
    index.ensure_current(db.knowledge)
    index.advance(db.events, 2)

    class Stuck(SpyBackend):
        def indexed(self):
            return 0

    spy = TxnSpy(db)
    backend = Stuck(db.vectors, spy)
    with pytest.raises(CursorConflict):
        Index(backend, HashEmbedder(), 0.9, transaction=spy.transaction).advance(db.events, 2)
    assert backend.log == []


def test_nearest_member_runs_outside_the_write_transaction(opened):
    db = appended(opened(), ev("e1"), ev("e2"), ev("e3", "other words"), ev("e4"))
    index, _, backend, _ = spied(db)

    index.advance(db.events, 4)

    assert [name for name, _ in backend.calls].count("nearest_member") == 4
    assert backend.calls and not any(inside for _, inside in backend.calls)
    assert reader(db, index).group("e4").members == ("e1", "e2", "e4")


def test_sync_knowledge_touched_only(opened):
    db = opened()
    for k in [item("k1", body="first"), item("k2", body="second"), item("k3", body="third")]:
        db.knowledge.put(k)
    index, emb, _, _ = spied(db)

    index.sync_knowledge(db.knowledge, {"k2"})

    assert emb.texts == ["deploy\nsecond"]
    assert emb.inside == []
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k2"]
    assert db.vectors.knowledge_digests() == {"k2": text_digest("deploy\nsecond")}
    index.sync_knowledge(db.knowledge, {"k2"})  # unchanged: nothing to embed or write
    assert len(emb.calls) == 1

    index.sync_knowledge(db.knowledge)  # full: the others too
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1", "k2", "k3"]
    assert sorted(emb.texts) == ["deploy\nfirst", "deploy\nsecond", "deploy\nthird"]
    assert emb.inside == []


def test_sync_knowledge_removes_vectors_of_demoted_ids(opened):
    db = opened()
    for k in [item("k1"), item("k2", body="kept")]:
        db.knowledge.put(k)
    index, _, _, _ = spied(db)
    index.sync_knowledge(db.knowledge)
    db.knowledge.remove("k1")

    index.sync_knowledge(db.knowledge, {"k1"})

    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k2"]
    # A full comparison removes the vector of an id the knowledge base lost
    # meanwhile, and an old vector stored without a digest.
    db.knowledge.remove("k2")
    db.vectors.put_vector("knowledge", "old", db.vectors.vector("knowledge", "k2"))
    index.sync_knowledge(db.knowledge)
    assert list(db.vectors.vectors("knowledge")) == []
    assert db.vectors.knowledge_digests() == {}


def test_sync_knowledge_skips_a_write_when_the_item_changed_meanwhile(opened):
    db = opened()
    db.knowledge.put(item("k1", body="old words"))
    other = opened()

    def meanwhile(texts, kind):
        if "deploy\nold words" in texts:
            other.knowledge.put(item("k1", body="new words"))  # another process replaces the item

    index, emb, backend, _ = spied(db, SpyEmbedder(None, during=meanwhile))

    index.sync_knowledge(db.knowledge, {"k1"})

    assert db.vectors.vector("knowledge", "k1") is None  # the old text's vector was not written
    assert backend.log == []
    index.sync_knowledge(db.knowledge, {"k1"})  # what the other writer's sync does
    assert db.vectors.knowledge_digests() == {"k1": text_digest("deploy\nnew words")}
    assert db.vectors.vector("knowledge", "k1") == db.vectors.cached(emb.model, text_digest("deploy\nnew words"))


def test_ensure_current_embeds_outside_the_transaction(opened):
    db = opened()
    db.knowledge.put(item("k1", body="first"))
    db.knowledge.put(item("k2", body="second"))
    index, emb, _, _ = spied(db, built=False)

    assert index.ensure_current(db.knowledge) is True

    assert sorted(emb.texts) == ["deploy\nfirst", "deploy\nsecond"]
    assert emb.inside == []
    assert not index.stale()
    assert db.vectors.generation() == 1
    assert set(db.vectors.knowledge_digests()) == {"k1", "k2"}
    assert index.ensure_current(db.knowledge) is False
    assert db.vectors.generation() == 1


def test_ensure_current_failure_changes_nothing(opened):
    db = appended(opened(), ev("e1"), ev("e2", "other words"))
    old = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
    old.ensure_current(db.knowledge)
    old.advance(db.events, 2)
    db.knowledge.put(item("k1", body="never embedded"))

    def state():
        v = db.vectors
        return v.meta(), v.generation(), v.indexed(), list(v.vectors("event")), list(v.vectors("knowledge"))

    before = state()
    spy = TxnSpy(db)
    fresh = Index(db.vectors, SpyEmbedder(spy, fail_on={"deploy\nnever embedded"}), 0.5, transaction=spy.transaction)

    with pytest.raises(EmbeddingError):
        fresh.ensure_current(db.knowledge)

    assert fresh.stale()  # the next sync tries again
    assert state() == before


# ── fix round 1: another config's generation, outage calls ─────────────────


def test_advance_refuses_to_write_into_an_index_of_another_config(opened):
    """Another process with another model rebuilds the index while this one
    embeds: this one must not write its vectors into that generation."""
    db_a, db_b = opened(), opened()
    appended(db_a, ev("e1", "one"), ev("e2", "two"))
    other = Index(db_b.vectors, HashEmbedder(dim=128), 0.9, transaction=db_b.transaction)

    def meanwhile(texts, kind):
        other.ensure_current(db_b.knowledge)

    first, emb, backend, _ = spied(db_a, SpyEmbedder(None, during=meanwhile))

    with pytest.raises(IndexConfigChanged):
        first.advance(db_a.events, 2)

    assert emb.calls  # it did embed, then found the other config
    assert db_a.vectors.meta() == other.want()
    assert list(db_a.vectors.vectors("event")) == [] and db_a.vectors.indexed() == 0
    assert backend.log == []
    other.advance(db_b.events, 2)
    assert {len(v) for _, v in db_a.vectors.vectors("event")} == {128}


def test_advance_refuses_an_index_never_built_for_it(opened):
    db = appended(opened(), ev("e1"))
    with pytest.raises(IndexConfigChanged):
        Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction).advance(db.events, 1)
    assert db.vectors.indexed() == 0


def test_a_failing_batch_of_one_text_is_not_tried_again(opened):
    db = appended(opened(), ev("e1", "fine"), ev("e2", "boom"))
    index, emb, _, _ = spied(db)
    index.advance(db.events, 1)
    emb.fail_on.add("boom")

    with pytest.raises(EmbeddingError):
        index.advance(db.events, 2)

    assert emb.batches[1:] == [["boom"]]  # one call: it is the culprit already


def test_advance_does_not_raise_when_every_text_succeeds_on_its_own(opened):
    db = appended(opened(), ev("e1", "one"), ev("e2", "two"), ev("e3", "three"))
    failed = []

    def flaky(texts, kind):
        if not failed:
            failed.append(texts)
            raise EmbeddingError("a hiccup")

    index, _, _, _ = spied(db, SpyEmbedder(None, during=flaky))

    index.advance(db.events, 3)  # the batch failed once, every text then succeeded

    assert failed == [["one", "two", "three"]]
    assert db.vectors.indexed() == 3


def test_advance_does_not_raise_when_another_process_indexed_past_the_failure(opened):
    db_a, db_b = opened(), opened()
    appended(db_a, ev("e1", "fine"), ev("e2", "boom"), ev("e3", "also fine"))
    second, _, b_backend, _ = spied(db_b)

    def meanwhile(texts, kind):
        if texts == ["boom"]:
            second.advance(db_b.events, 3)  # its embedder works

    first, _, a_backend, _ = spied(db_a, SpyEmbedder(None, fail_on={"boom"}, during=meanwhile))

    first.advance(db_a.events, 3)

    assert db_a.vectors.indexed() == 3
    assert a_backend.log == []  # the other process indexed every event, once
    assert [e[-1] for e in b_backend.log] == ["e1", "e1", "e2", "e2", "e3", "e3"]
