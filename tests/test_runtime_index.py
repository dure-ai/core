"""The runtime with an embedding index: each event is indexed in the same
transaction before it is delivered, engines read similarity through
ctx.similar, and a changed index version replays the engine (spec §7)."""

from datetime import UTC, datetime

import pytest

from cortex_mem.contract import (
    Caller,
    Demote,
    Event,
    Group,
    Handoff,
    Knowledge,
    Promote,
)
from cortex_mem.embed import EmbeddingError, HashEmbedder
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase
from cortex_mem.store.sqlite import SqliteDatabase

T0 = datetime(2026, 11, 15, tzinfo=UTC)


def ev(id, text="deploy uses tags", kind="record", user="u1", team="t1", scope="personal", **kw):
    return Event(id=id, kind=kind, at=T0, user=user, team=team, session="s1", scope=scope, text=text, **kw)


def caller(user="u1", team="t1"):
    return Caller(user=user, team=team, session="s2", at=T0)


def item(id, sources, scope="personal", owner="u1", title="deploy", body="deploy uses tags"):
    return Knowledge(id=id, kind="knowledge", title=title, body=body, scope=scope, owner=owner, sources=sources)


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


class ProbeEngine:
    """Records what ctx.similar says while observing; decisions and handoffs
    are set per event id by the test."""

    def __init__(self, decisions=None, handoffs=(), on_tick=(), ask_knowledge=False):
        self.decisions = decisions or {}
        self.handoffs = list(handoffs)
        self.on_tick = list(on_tick)
        self.ask_knowledge = ask_knowledge  # a query embeds, so only when a test wants it
        self.seen = []
        self.groups = {}  # event id -> ctx.similar.group(event.id) while observing it
        self.similar = {}  # call -> the ctx.similar it was given
        self.knowledge_hits = {}  # event id -> ctx.similar.knowledge(event.text) ids

    def observe(self, event, ctx):
        self.seen.append(event.id)
        self.similar["observe"] = ctx.similar
        if ctx.similar is not None:
            self.groups[event.id] = ctx.similar.group(event.id)
        if ctx.similar is not None and self.ask_knowledge:
            hits = ctx.similar.knowledge(event.text, caller(), "org")
            self.knowledge_hits[event.id] = [k.id for k, _ in hits]
        ctx.state.put("n", (ctx.state.get("n") or 0) + 1)
        return self.decisions.get(event.id, [])

    def tick(self, now, ctx):
        self.similar["tick"] = ctx.similar
        return self.on_tick

    def handoff(self, query, caller, scope, ctx, k=6):
        self.similar["handoff"] = ctx.similar
        if ctx.similar is not None:
            hits = ctx.similar.search(query, caller, scope)
            return [Handoff(e.id, e.text, (e.id,)) for e, _ in hits] + self.handoffs
        return self.handoffs


def make_db(kind, tmp_path):
    return InMemoryDatabase() if kind == "memory" else SqliteDatabase(tmp_path / "core.db")


def runtime(engine, db, embedder=None, threshold=0.9, max_failures=None, with_index=True):
    index = Index(db.vectors, embedder or HashEmbedder(), threshold) if with_index else None
    return Runtime(
        engine, db.events, db.knowledge, db.state("probe"), db.transaction, max_failures, index=index
    )


def snapshot(db):
    """Everything the index wrote."""
    v = db.vectors
    events = list(v.vectors("event"))
    return (
        v.meta(),
        v.indexed(),
        events,
        list(v.vectors("knowledge")),
        {id: v.group_of(id) for id, _ in events},
    )


# ── without an index ────────────────────────────────────────────────────────


def scenario(with_index):
    """Promote, a refused promote, a demote, a tick decision and a filtered handoff."""
    engine = ProbeEngine(
        decisions={
            "e1": [Promote(item("k1", ("e1",)))],
            "e2": [
                Promote(item("k2", ("e2",), scope="team", owner="t1", body="payments need vendor v2")),
                Promote(item("bad", ("e1",), scope="team", owner="t1")),  # personal source
            ],
            "e3": [Demote("k1", "failed", ("e3",))],
        },
        handoffs=[Handoff("k2", "payments", ("e2",)), Handoff("hidden", "other team", ("x1",))],
        on_tick=[Promote(item("k3", ("e2",), scope="team", owner="t1", body="tick item"))],
    )
    db = InMemoryDatabase()
    rt = runtime(engine, db, with_index=with_index)
    rt.append(ev("e1"))
    rt.append(ev("e2", "payments need vendor v2", scope="team"))
    rt.append(ev("e3", "deploy tags broke", kind="outcome", outcome="failure", refs=("k1",)))
    rt.append(ev("x1", "other team secret", user="u9", team="t9", scope="team"))
    rt.tick(T0)
    handed = rt.handoff("payments", caller(), "team")
    result = (
        [(i.id, i.scope, i.owner) for i in db.knowledge.all()],
        [(type(v.decision).__name__, v.accepted, v.reason) for v in rt.verdicts],
        rt.state.cursor(),
        rt.state.get("n"),
        engine.seen,
    )
    return engine, db, result, handed


def test_without_an_index_nothing_changes():
    engine, db, plain, handed = scenario(with_index=False)
    assert set(engine.similar) == {"observe", "tick", "handoff"}
    assert all(s is None for s in engine.similar.values())
    assert db.vectors.meta() is None and db.vectors.indexed() == 0  # never touched
    assert [h.id for h in handed] == ["k2"]
    assert plain[0] == [("k2", "team", "t1"), ("k3", "team", "t1")]

    # The same scenario with an index decides and applies exactly the same.
    _, _, indexed, _ = scenario(with_index=True)
    assert indexed == plain


# ── delivery ────────────────────────────────────────────────────────────────


def test_engine_sees_the_group_of_the_event_it_observes():
    engine = ProbeEngine()
    rt = runtime(engine, InMemoryDatabase())
    rt.append(ev("e1"))
    rt.append(ev("e2"))  # same words, same (scope, owner)
    rt.append(ev("e3", "deploy tags broke", kind="outcome", outcome="failure", refs=("e1",)))

    assert engine.groups["e1"] == Group("e1", "personal", "u1", ("e1",))
    assert engine.groups["e2"] == Group("e1", "personal", "u1", ("e1", "e2"))
    assert engine.groups["e3"] is None  # outcomes are not embedded


def test_handoff_context_has_similarity_and_is_read_only():
    engine = ProbeEngine()
    db = InMemoryDatabase()
    rt = runtime(engine, db)
    rt.append(ev("e1", "deploy uses tags"))
    rt.append(ev("e2", "payments need vendor v2"))
    rt.append(ev("x1", "deploy uses tags", user="u9"))  # someone else's personal record
    before = snapshot(db)

    handed = rt.handoff("deploy uses tags", caller(), "personal")

    similar = engine.similar["handoff"]
    assert similar is not None
    assert {name for name in dir(similar) if not name.startswith("_")} == {
        "model", "search", "neighbors", "knowledge", "group",
    }
    assert handed[0].id == "e1" and "x1" not in [h.id for h in handed]
    assert snapshot(db) == before  # the query vector is not stored
    rt.tick(T0)
    assert engine.similar["tick"] is not None


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_embedding_failure_delivers_nothing_and_retries(store, tmp_path):
    db = make_db(store, tmp_path)
    emb = FailingEmbedder({"boom"})
    engine = ProbeEngine(decisions={id: [Promote(item(f"k-{id}", (id,)))] for id in ["e1", "e2"]})
    rt = runtime(engine, db, emb, max_failures=1)
    rt.append(ev("e1", "fine"))
    before = (snapshot(db), rt.state.cursor(), rt.state.get("n"), [i.id for i in db.knowledge.all()])

    with pytest.raises(EmbeddingError):
        rt.append(ev("e2", "boom"))
    with pytest.raises(EmbeddingError):  # a second try is not counted toward max_failures either
        rt.sync()

    assert engine.seen == ["e1"]
    assert (snapshot(db), rt.state.cursor(), rt.state.get("n"), [i.id for i in db.knowledge.all()]) == before
    assert db.vectors.group_of("e2") is None
    assert rt.skipped == []
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]  # the event stays in the log

    emb.fail_on.clear()  # the embedder recovers
    assert rt.sync() == 1
    assert engine.seen == ["e1", "e2"]
    assert engine.groups["e2"] == Group("e2", "personal", "u1", ("e2",))
    assert (db.vectors.indexed(), rt.state.cursor(), rt.state.get("n")) == (2, 2, 2)
    assert [i.id for i in db.knowledge.all()] == ["k-e1", "k-e2"]
    assert rt.skipped == []


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_promoted_item_embedding_failure_commits_nothing_and_retries(store, tmp_path):
    db = make_db(store, tmp_path)
    emb = FailingEmbedder({"deploy\nboom item"})  # the promoted item's passage, not the event
    engine = ProbeEngine(
        decisions={
            "e1": [Promote(item("k1", ("e1",)))],
            "e2": [Promote(item("k2", ("e2",), body="boom item"))],
        }
    )
    rt = runtime(engine, db, emb, max_failures=1)
    rt.append(ev("e1", "fine"))

    def observed():
        return (
            rt.state.cursor(),
            rt.state.get("n"),
            [i.id for i in db.knowledge.all()],
            list(db.vectors.vectors("knowledge")),
            [(type(v.decision).__name__, v.accepted) for v in rt.verdicts],
        )

    before = observed()
    with pytest.raises(EmbeddingError):
        rt.append(ev("e2", "fine too"))
    with pytest.raises(EmbeddingError):  # not counted toward max_failures
        rt.sync()
    assert observed() == before
    assert rt.skipped == []

    emb.fail_on.clear()
    assert rt.sync() == 1
    assert (rt.state.cursor(), rt.state.get("n")) == (2, 2)
    assert [i.id for i in db.knowledge.all()] == ["k1", "k2"]
    assert db.vectors.vector("knowledge", "k2") is not None
    assert [v.accepted for v in rt.verdicts] == [True, True]
    assert rt.skipped == []


# ── index version ───────────────────────────────────────────────────────────


TABLE = {"alpha": [1, 0, 0], "alpha-ish": [0.7, 0.714, 0], "gamma": [0, 0, 1], "deploy\nalpha": [1, 0, 0]}


def test_changed_threshold_rebuilds_groups_and_replays_the_engine(tmp_path):
    path = tmp_path / "core.db"
    first = ProbeEngine(decisions={"e1": [Promote(item("k1", ("e1",), body="alpha"))]})
    db = SqliteDatabase(path)
    rt = runtime(first, db, TableEmbedder(TABLE), threshold=0.9)
    for id, text in [("e1", "alpha"), ("e2", "alpha-ish"), ("e3", "gamma")]:
        rt.append(ev(id, text))
    assert first.groups["e2"].id == "e2"  # cosine 0.7 < 0.9
    db.con.close()

    # Same threshold: nothing to rebuild, nothing replayed.
    db = SqliteDatabase(path)
    again = ProbeEngine()
    runtime(again, db, TableEmbedder(TABLE), threshold=0.9).sync()
    assert again.seen == []
    db.con.close()

    db = SqliteDatabase(path)
    second = ProbeEngine()
    emb = TableEmbedder(TABLE)
    rt = runtime(second, db, emb, threshold=0.5)
    assert rt.sync() == 3

    assert second.seen == ["e1", "e2", "e3"]  # the engine got the whole log again
    assert second.groups["e2"] == Group("e1", "personal", "u1", ("e1", "e2"))
    assert second.groups["e3"].id == "e3"
    assert db.vectors.meta() == {"model": "table", "group_threshold": 0.5}
    assert rt.state.get("n") == 3  # state was reset, not added to
    assert rt.state.version() == ""  # no state_version on the engine, none stored
    assert [i.id for i in db.knowledge.all()] == ["k1"]  # knowledge is kept
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1"]
    assert emb.calls == []  # every passage came from the cache


def test_rebuild_keeps_the_engines_state_version():
    class Versioned(ProbeEngine):
        state_version = "v7"

    db = InMemoryDatabase()
    runtime(Versioned(), db, threshold=0.9).append(ev("e1"))
    engine = Versioned()
    rt = runtime(engine, db, threshold=0.5)
    rt.sync()
    assert engine.seen == ["e1"]
    assert rt.state.version() == "v7"


# ── knowledge vectors and cursors ───────────────────────────────────────────


def test_knowledge_vectors_follow_promote_and_demote():
    engine = ProbeEngine(
        decisions={
            "e1": [Promote(item("k1", ("e1",)))],
            "e3": [Demote("k1", "failed", ("e3",))],
        },
        ask_knowledge=True,
    )
    db = InMemoryDatabase()
    rt = runtime(engine, db)
    rt.append(ev("e1", "deploy uses tags"))
    rt.append(ev("e2", "deploy uses tags"))
    assert engine.knowledge_hits["e2"] == ["k1"]
    assert db.vectors.vector("knowledge", "k1") is not None

    rt.append(ev("e3", "deploy tags broke", kind="outcome", outcome="failure", refs=("k1",)))
    rt.append(ev("e4", "deploy uses tags"))
    assert engine.knowledge_hits["e4"] == []
    assert db.vectors.vector("knowledge", "k1") is None


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_index_and_engine_cursors_move_together(store, tmp_path):
    class Breaks(ProbeEngine):
        def observe(self, event, ctx):
            if event.id == "bad":
                raise RuntimeError("boom")
            return super().observe(event, ctx)

    db = make_db(store, tmp_path)
    rt = runtime(Breaks(), db, max_failures=1)
    rt.append(ev("e1"))
    assert db.vectors.indexed() == rt.state.cursor() == 1
    rt.append(ev("e2", "done", kind="outcome", outcome="success", refs=("e1",)))
    assert db.vectors.indexed() == rt.state.cursor() == 2
    rt.append(ev("bad", "skipped after one failure"))
    assert [s.event_id for s in rt.skipped] == ["bad"]
    assert db.vectors.indexed() == rt.state.cursor() == 3
    rt.append(ev("e4", "more words"))
    assert db.vectors.indexed() == rt.state.cursor() == 4
