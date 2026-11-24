"""The runtime with an embedding index: each event is indexed before it is
observed (the embedder and the engine never run inside a write transaction),
engines read similarity through ctx.similar, knowledge vectors follow the
knowledge base after each decision commits, a rebuilt index replays the
engine (spec §7), and index settings other than the store's are reported,
never rebuilt over."""

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from spy import SpyEmbedder, TxnSpy

from cortex_mem.contract import (
    Caller,
    Demote,
    Event,
    Group,
    Handoff,
    Knowledge,
    Promote,
)
from cortex_mem.embed import EmbeddingError, EmbeddingRejected, HashEmbedder
from cortex_mem.index import Index, IndexConfigChanged, IndexMismatch, drop_index
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


def reopener(store, tmp_path, request):
    """A zero-argument opener: each call is one more process on the same storage.
    "postgres" runs only when CORTEX_TEST_PG_URL is set."""
    if store == "memory":
        shared = InMemoryDatabase()
        return lambda: shared
    if store == "sqlite":
        return lambda: SqliteDatabase(tmp_path / "core.db")
    url = request.getfixturevalue("pg_schema")
    from cortex_mem.store.postgres import PostgresDatabase

    opened = []

    def open_pg():
        db = PostgresDatabase(url)
        opened.append(db)
        return db

    request.addfinalizer(lambda: [db.close() for db in opened])
    return open_pg


STORES = ["memory", "sqlite", "postgres"]


def runtime(engine, db, embedder=None, threshold=0.9, max_failures=None, with_index=True, name="probe"):
    transaction = getattr(embedder, "spy", None)
    transaction = db.transaction if transaction is None else transaction.transaction
    index = Index(db.vectors, embedder or HashEmbedder(), threshold, transaction=transaction) if with_index else None
    return Runtime(
        engine, db.events, db.knowledge, db.state(name), transaction, max_failures, index=index
    )


def rebuild(db, embedder=None, threshold=0.9):
    """The explicit rebuild, as one command run by a person does it."""
    transaction = getattr(embedder, "spy", None)
    transaction = db.transaction if transaction is None else transaction.transaction
    return Index(db.vectors, embedder or HashEmbedder(), threshold, transaction=transaction).rebuild(db.knowledge)


def unindexed_runtime(engine, db, name="probe"):
    """A process with no embedder, as the server builds it: it is given the
    store's vector backend so that it can see an index it does not use."""
    return Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction, vectors=db.vectors)


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
def test_promotion_commits_even_if_its_embedding_fails(store, tmp_path):
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

    with pytest.raises(EmbeddingError):
        rt.append(ev("e2", "fine too"))

    # The decision is not held back by its embedding: the event is delivered and the item promoted.
    assert (rt.state.cursor(), rt.state.get("n")) == (2, 2)
    assert [i.id for i in db.knowledge.all()] == ["k1", "k2"]
    assert [(type(v.decision).__name__, v.accepted) for v in rt.verdicts] == [("Promote", True)] * 2
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1"]
    with pytest.raises(EmbeddingError):  # still down: the pending item is tried first, and raised
        rt.sync()
    assert engine.seen == ["e1", "e2"]
    assert rt.skipped == []

    emb.fail_on.clear()  # the embedder recovers
    assert rt.sync() == 0
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1", "k2"]
    assert engine.seen == ["e1", "e2"]


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_first_sync_reconciles_all_knowledge(store, tmp_path):
    db = make_db(store, tmp_path)
    runtime(ProbeEngine(), db).sync()  # the index is built
    # A process promoted these and died before it embedded them.
    db.knowledge.put(item("k1", ("e1",), body="lost one"))
    db.knowledge.put(item("k2", ("e1",), body="lost two"))
    db.vectors.put_vector("knowledge", "gone", [1.0] * 256, "stale")  # demoted meanwhile
    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["gone"]

    emb = CountingEmbedder()
    rt = runtime(ProbeEngine(), db, emb)
    rt.sync()

    assert [id for id, _ in db.vectors.vectors("knowledge")] == ["k1", "k2"]
    assert sorted(t for t, _ in emb.calls) == ["deploy\nlost one", "deploy\nlost two"]
    # Only the first sync compares everything; later ones look at what they changed.
    db.knowledge.put(item("k3", ("e1",), body="lost three"))
    rt.sync()
    assert db.vectors.vector("knowledge", "k3") is None


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_the_runtime_never_embeds_inside_a_write_transaction(store, tmp_path):
    db = make_db(store, tmp_path)
    emb = SpyEmbedder(TxnSpy(db))
    engine = ProbeEngine(
        decisions={"e1": [Promote(item("k1", ("e1",)))], "e3": [Demote("k1", "failed", ("e3",))]},
        on_tick=[Promote(item("k2", ("e2",), body="tick item"))],
    )
    rt = runtime(engine, db, emb)
    rt.append(ev("e1"))
    rt.append(ev("e2", "payments need vendor v2"))
    db.events.append(ev("e5", "appended by another process"))
    rt.append(ev("e3", "deploy tags broke", kind="outcome", outcome="failure", refs=("k1",)))
    rt.tick(T0)
    # A rebuild with a new threshold, and the replay of the engine that follows it.
    rebuilding = SpyEmbedder(TxnSpy(db))
    db.knowledge.put(item("k9", ("e1",), body="only the rebuild embeds this"))
    assert rebuild(db, rebuilding, threshold=0.5) is True
    runtime(ProbeEngine(), db, rebuilding, threshold=0.5).sync()

    assert emb.calls and emb.inside == []
    assert rebuilding.calls and rebuilding.inside == []
    assert engine.seen == ["e1", "e2", "e5", "e3"]
    # The rebuild reset the engine and emptied the knowledge base; this replay
    # promotes nothing, and a replay does not tick (k2).
    assert db.knowledge.all() == [] and list(db.vectors.vectors("knowledge")) == []


# ── index version ───────────────────────────────────────────────────────────


TABLE = {"alpha": [1, 0, 0], "alpha-ish": [0.7, 0.714, 0], "gamma": [0, 0, 1], "deploy\nalpha": [1, 0, 0]}


def test_an_explicit_rebuild_with_a_new_threshold_regroups_and_replays_the_engine(tmp_path):
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
    with pytest.raises(IndexMismatch):  # another threshold alone rebuilds nothing
        rt.sync()
    assert second.seen == [] and db.vectors.meta()["group_threshold"] == 0.9
    assert rebuild(db, emb, threshold=0.5) is True
    assert rt.sync() == 3

    assert second.seen == ["e1", "e2", "e3"]  # the engine got the whole log again
    assert second.groups["e2"] == Group("e1", "personal", "u1", ("e1", "e2"))
    assert second.groups["e3"].id == "e3"
    assert db.vectors.meta() == {"model": "table", "group_threshold": 0.5, "store": "sqlite"}
    assert rt.state.get("n") == 3  # state was reset, not added to
    # No state_version on the engine: the version records only the index it was built on.
    assert json.loads(rt.state.version()) == [None, fingerprint(db, 0.5, model="table")]
    # The reset cleared the knowledge base: the replay decides it again, and this
    # engine promotes nothing, so k1 (promoted under the old grouping) is gone.
    assert db.knowledge.all() == []
    assert list(db.vectors.vectors("knowledge")) == []
    assert emb.calls == []  # nothing new to embed


def test_rebuild_keeps_the_engines_state_version():
    class Versioned(ProbeEngine):
        state_version = "v7"

    db = InMemoryDatabase()
    runtime(Versioned(), db, threshold=0.9).append(ev("e1"))
    rebuild(db, threshold=0.5)
    engine = Versioned()
    rt = runtime(engine, db, threshold=0.5)
    rt.sync()
    assert engine.seen == ["e1"]
    assert json.loads(rt.state.version()) == ["v7", fingerprint(db, 0.5)]


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


# ── embedding outside SQLite's write lock ───────────────────────────────────


class BatchCountingEmbedder(CountingEmbedder):
    def __init__(self, inner=None):
        super().__init__(inner)
        self.batches = []

    def embed(self, texts, kind):
        self.batches.append(list(texts))
        return super().embed(texts, kind)


class SlowEmbedder(CountingEmbedder):
    """Takes `delay` seconds per call and says when the first call began."""

    def __init__(self, delay, model="slow"):
        super().__init__()
        self.model = model
        self.delay = delay
        self.started = threading.Event()

    def embed(self, texts, kind):
        self.started.set()
        time.sleep(self.delay)
        return super().embed(texts, kind)


def test_sync_embeds_the_pending_passages_in_one_batch():
    db = InMemoryDatabase()
    emb = BatchCountingEmbedder()
    engine = ProbeEngine()
    rt = runtime(engine, db, emb)
    rt.sync()  # builds the empty index
    # Appended by other processes, not yet delivered here.
    for i in range(5):
        db.events.append(ev(f"e{i}", f"fact number {i}"))
    db.events.append(ev("o1", "done", kind="outcome", outcome="success", refs=("e0",)))
    db.events.append(ev("e5", "fact number 0"))  # a text seen before

    assert rt.sync() == 7

    assert emb.batches == [[f"fact number {i}" for i in range(5)]]
    assert engine.groups["e5"].members == ("e0", "e5")


@pytest.mark.parametrize("phase", ["events", "rebuild"])
def test_another_connection_can_append_while_core_embeds(phase, tmp_path):
    path = tmp_path / "core.db"
    db = SqliteDatabase(path)
    emb = SlowEmbedder(delay=0.5)
    engine = ProbeEngine()
    if phase == "events":
        rt = runtime(engine, db, emb)
        rt.sync()  # index built, nothing to embed yet
        db.events.append(ev("e1", "slow words"))
    else:
        db.knowledge.put(item("k1", ("e0",), body="slow knowledge"))  # the new model must embed it
        rt = runtime(engine, db, emb)
    other = SqliteDatabase(path)
    other.con.execute("PRAGMA busy_timeout = 50")  # far shorter than one embedding
    errors = []

    def deliver():
        try:
            rt.sync()
        except (sqlite3.Error, EmbeddingError) as exc:  # reported below; anything else fails loudly
            errors.append(exc)

    worker = threading.Thread(target=deliver)
    worker.start()
    assert emb.started.wait(5)
    try:
        other.events.append(ev("x1", "appended meanwhile", user="u2"))  # raised "database is locked" before
    finally:
        worker.join(10)
    assert errors == []
    assert [e.id for _, e in other.events.since(0)][-1] == "x1"
    rt.sync()
    assert "x1" in engine.seen


def test_a_failing_text_in_a_batch_stops_only_its_own_event():
    db = InMemoryDatabase()
    emb = FailingEmbedder({"boom"})  # fails any batch that contains it
    engine = ProbeEngine()
    rt = runtime(engine, db, emb)
    rt.sync()
    for e in [ev("e1", "fine"), ev("e2", "boom"), ev("e3", "fine too")]:
        db.events.append(e)

    with pytest.raises(EmbeddingError):
        rt.sync()

    assert engine.seen == ["e1"]  # delivered although its batch failed
    assert (rt.state.cursor(), db.vectors.indexed(), rt.skipped) == (1, 1, [])
    emb.fail_on.clear()
    assert rt.sync() == 2


class RefusingEmbedder(CountingEmbedder):
    """Refuses the given texts: alone with EmbeddingRejected, in a batch with
    EmbeddingError (as the Upstage embedder does)."""

    def __init__(self, refuse):
        super().__init__()
        self.refuse = set(refuse)

    def embed(self, texts, kind):
        if any(t in self.refuse for t in texts):
            if len(texts) == 1:
                raise EmbeddingRejected("the embedding API refused this text")
            raise EmbeddingError("the embedding API refused the request")
        return super().embed(texts, kind)


def test_a_refused_text_is_delivered_without_a_group_and_holds_nothing_back():
    db = InMemoryDatabase()
    engine = ProbeEngine()
    rt = runtime(engine, db, RefusingEmbedder({"huge"}))
    rt.sync()
    for e in [ev("e1", "fine"), ev("e2", "huge"), ev("e3", "fine too")]:
        db.events.append(e)

    assert rt.sync() == 3
    assert engine.seen == ["e1", "e2", "e3"]
    assert engine.groups["e2"] is None and engine.groups["e3"] is not None
    assert (rt.state.cursor(), db.vectors.indexed(), rt.skipped) == (3, 3, [])
    assert rt.handoff("huge", caller(), "personal") == []  # a refused query finds nothing


# ── every engine on one store follows the index version ─────────────────────


@pytest.mark.parametrize("store", STORES)
def test_every_engine_on_the_store_is_replayed_after_the_index_changes(store, tmp_path, request):
    reopen = reopener(store, tmp_path, request)  # each call: a new process on the same storage

    a_old, b_old = ProbeEngine(), ProbeEngine()
    rt_a = runtime(a_old, reopen(), TableEmbedder(TABLE), threshold=0.9, name="a")
    rt_b = runtime(b_old, reopen(), TableEmbedder(TABLE), threshold=0.9, name="b")
    for id, text in [("e1", "alpha"), ("e2", "alpha-ish"), ("e3", "gamma")]:
        rt_a.append(ev(id, text))
    rt_b.sync()
    assert a_old.groups["e2"].members == b_old.groups["e2"].members == ("e2",)  # 0.7 < 0.9

    # The index is rebuilt with a lower threshold, and engine a restarts with it: a is replayed.
    rebuild(reopen(), TableEmbedder(TABLE), threshold=0.5)
    a_new = ProbeEngine()
    rt_a = runtime(a_new, reopen(), TableEmbedder(TABLE), threshold=0.5, name="a")
    assert rt_a.sync() == 3
    assert a_new.groups["e2"] == Group("e1", "personal", "u1", ("e1", "e2"))

    # Engine b restarts with the same new config: the index is already current,
    # but b's state was built on the old groups, so b is replayed too.
    b_new = ProbeEngine()
    emb = TableEmbedder(TABLE)
    rt_b = runtime(b_new, reopen(), emb, threshold=0.5, name="b")
    assert rt_b.sync() == 3
    assert b_new.seen == ["e1", "e2", "e3"]
    assert b_new.groups["e2"] == Group("e1", "personal", "u1", ("e1", "e2"))
    assert (rt_b.state.cursor(), rt_b.state.get("n")) == (3, 3)  # reset, not added to
    assert emb.calls == []

    # Both are current now: nothing is replayed again.
    again = ProbeEngine()
    assert runtime(again, reopen(), TableEmbedder(TABLE), threshold=0.5, name="b").sync() == 0
    assert again.seen == []


def test_the_index_is_checked_on_every_sync_not_once_per_process():
    db = InMemoryDatabase()
    engine = ProbeEngine()
    rt = runtime(engine, db, TableEmbedder(TABLE), threshold=0.9, name="a")
    for id, text in [("e1", "alpha"), ("e2", "alpha-ish")]:
        rt.append(ev(id, text))
    # The index is rebuilt with a new threshold while this runtime keeps running.
    rebuild(db, TableEmbedder(TABLE), threshold=0.5)
    runtime(ProbeEngine(), db, TableEmbedder(TABLE), threshold=0.5, name="other").sync()
    engine.seen.clear()
    state = (rt.state.position(), dict(rt.state.scan()))
    index = db.vectors.position()

    with pytest.raises(IndexMismatch) as raised:
        rt.sync()

    stored = {"model": "table", "group_threshold": 0.5, "store": db.vectors.kind}
    assert raised.value.stored == stored
    assert raised.value.current == {**stored, "group_threshold": 0.9}
    assert "0.5" in str(raised.value) and "0.9" in str(raised.value)
    # Reported, not rebuilt back: the index and this engine's state are as they were.
    assert engine.seen == []
    assert db.vectors.position() == index
    assert (rt.state.position(), dict(rt.state.scan())) == state


# ── one index configuration per store: reported, never rebuilt over ────────


class Counting(ProbeEngine):
    """Counts every observe call across the processes of one test."""

    calls = 0

    def observe(self, event, ctx):
        type(self).calls += 1
        return super().observe(event, ctx)


def _client(reopen, kind):
    """A fresh client process with other index settings than the store's."""
    if kind == "threshold":
        return runtime(Counting(), reopen(), threshold=0.8)
    return unindexed_runtime(Counting(), reopen())


@pytest.mark.parametrize("client", ["threshold", "no embedder"])
@pytest.mark.parametrize("store", STORES)
def test_two_processes_with_different_index_settings_do_not_replay_each_other(
    store, client, tmp_path, request, monkeypatch
):
    """A long-running service, and fresh clients with another grouping
    threshold or with embedding off, take turns on one store. Each client
    used to reset the engine and replay the whole log, and the service
    replayed it back on its next call."""
    monkeypatch.setattr(Counting, "calls", 0)
    reopen = reopener(store, tmp_path, request)
    service = runtime(Counting(decisions={"e0": [Promote(item("k0", ("e0",), body="fact 0"))]}), reopen())
    for i in range(20):
        service.append(ev(f"e{i}", f"fact {i}"))
    db = reopen()
    assert Counting.calls == 20

    def stored():
        state = db.state("probe")
        return db.vectors.position(), state.position(), dict(state.scan()), db.knowledge.all()

    before = stored()
    for turn in range(3):
        fresh = _client(reopen, client)
        with pytest.raises(IndexMismatch):
            fresh.handoff("fact", caller(), "personal")
        assert fresh.engine.seen == []
        assert stored() == before  # no rebuild, no reset, nothing emptied
        assert service.handoff("fact", caller(), "personal") != []
        assert Counting.calls == 20 + turn  # the service observes only what is new
        service.append(ev(f"n{turn}", f"new {turn}"))
        before = stored()
    assert Counting.calls == 23
    assert db.vectors.generation() == 1


@pytest.mark.parametrize("store", STORES)
def test_a_rebuild_right_after_the_index_check_does_not_let_the_old_config_reset_the_engine(
    store, tmp_path, request
):
    """The index is rebuilt with another threshold, and replayed by a process
    with it, between this Runtime's index check and its version check: the
    version it then wants names its own settings, which the store no longer
    holds, so it must report the mismatch and not empty what the others built."""
    reopen = reopener(store, tmp_path, request)
    decisions = {"e1": [Promote(item("k1", ("e1",)))]}
    db = reopen()
    rt = runtime(ProbeEngine(decisions=decisions), db)
    rt.append(ev("e1"))
    real, ran = rt.index.ensure_current, []

    def checked_then_rebuilt(knowledge):
        current = real(knowledge)
        if not ran:
            ran.append(True)
            other = reopen()
            rebuild(other, threshold=0.8)
            assert runtime(ProbeEngine(decisions=decisions), other, threshold=0.8).sync() == 1
        return current

    rt.index.ensure_current = checked_then_rebuilt
    with pytest.raises(IndexMismatch):
        rt.sync()

    assert ran == [True]
    assert rt.state.cursor() == 1 and [k.id for k in db.knowledge.all()] == ["k1"]
    assert json.loads(rt.state.version())[1]["group_threshold"] == 0.8


class Meanwhile(ProbeEngine):
    """Runs `during` once, inside its first observe or tick: something another
    process does while this engine runs outside any transaction."""

    def __init__(self, during, fail=False, **kw):
        super().__init__(**kw)
        self.during, self.fail = during, fail

    def _once(self):
        during, self.during = self.during, None
        if during is not None:
            during()

    def observe(self, event, ctx):
        self._once()
        if self.fail:
            raise RuntimeError("the engine fails on this event")
        return super().observe(event, ctx)

    def tick(self, now, ctx):
        self._once()
        return super().tick(now, ctx)


@pytest.mark.parametrize("path", ["observe", "skip", "tick"])
@pytest.mark.parametrize("store", STORES)
def test_a_runtime_with_no_embedder_commits_nothing_once_the_store_has_an_index(store, path, tmp_path, request):
    """An index is built while a process with no embedder observes, gives up
    on a failing event, or ticks: its result must not land on a store whose
    index it does not read."""
    reopen = reopener(store, tmp_path, request)
    db = reopen()
    engine = Meanwhile(
        lambda: rebuild(reopen()),
        fail=path == "skip",
        decisions={"e1": [Promote(item("k1", ("e1",)))]},
        on_tick=[Promote(item("k2", ("e0",), body="tick item"))],
    )
    rt = Runtime(
        engine, db.events, db.knowledge, db.state("probe"), db.transaction,
        max_failures=1 if path == "skip" else None, vectors=db.vectors,
    )
    if path == "tick":
        engine.during, during = None, engine.during
        rt.append(ev("e0"))
        engine.during = during
        with pytest.raises(IndexMismatch):
            rt.tick(T0)
        assert rt.state.cursor() == 1
    else:
        db.events.append(ev("e1"))
        with pytest.raises(IndexMismatch):
            rt.sync()
        assert rt.state.cursor() == 0 and rt.skipped == []
    assert db.knowledge.all() == [] and db.vectors.generation() == 1


@pytest.mark.parametrize("threshold, reported", [(0.8, True), (0.9, False)], ids=["other settings", "same settings"])
@pytest.mark.parametrize("store", STORES)
def test_a_tick_over_an_index_rebuilt_meanwhile_reports_other_settings(store, threshold, reported, tmp_path, request):
    """The index is rebuilt while an indexed Runtime ticks: the result is
    dropped either way, and settings other than its own are reported by that
    call, not left for the next sync."""
    reopen = reopener(store, tmp_path, request)
    db = reopen()

    def rebuilt():
        other = reopen()
        other.vectors.reset({"model": "someone-else", "group_threshold": 0.1, "store": other.vectors.kind})
        rebuild(other, threshold=threshold)

    engine = Meanwhile(None, on_tick=[Promote(item("k2", ("e0",), body="tick item"))])
    rt = runtime(engine, db)
    rt.append(ev("e0"))
    engine.during = rebuilt
    if reported:
        with pytest.raises(IndexMismatch):
            rt.tick(T0)
    else:
        rt.tick(T0)
    assert db.knowledge.all() == [] and rt.state.cursor() == 1


def test_a_runtime_takes_an_index_or_the_vector_backend_not_both():
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9)
    with pytest.raises(ValueError, match="vectors"):
        Runtime(ProbeEngine(), db.events, db.knowledge, db.state("probe"), index=index, vectors=db.vectors)


def test_a_runtime_with_no_embedder_delivers_on_a_store_with_no_index():
    db = InMemoryDatabase()
    engine = ProbeEngine()
    rt = unindexed_runtime(engine, db)
    rt.append(ev("e1"))
    assert engine.seen == ["e1"] and db.vectors.position() == (None, 0, 0)


# ── the engine version records the index generation (spec §3.5) ─────────────


def fingerprint(db, threshold=0.9, model="hash-256"):
    return {"model": model, "group_threshold": threshold, "store": db.vectors.kind, "generation": db.vectors.generation()}


def composite(state_version, fp):
    return json.dumps([state_version, fp], sort_keys=True, separators=(",", ":"))


class ScanKeys(ProbeEngine):
    """Also records every key ctx.state.scan() shows while observing."""

    def __init__(self):
        super().__init__()
        self.keys = set()

    def observe(self, event, ctx):
        self.keys.update(k for k, _ in ctx.state.scan())
        decisions = super().observe(event, ctx)
        self.keys.update(k for k, _ in ctx.state.scan())
        return decisions


@pytest.mark.parametrize("store", STORES)
def test_changed_index_config_replays_every_engine_without_reserved_keys(store, tmp_path, request):
    reopen = reopener(store, tmp_path, request)
    first = reopen()
    spy_a, spy_b = SpyEmbedder(TxnSpy(first)), SpyEmbedder(TxnSpy(first))
    a, b = ScanKeys(), ScanKeys()
    rt_a = runtime(a, first, spy_a, threshold=0.9, name="a")
    rt_b = runtime(b, first, spy_b, threshold=0.9, name="b")
    for i in range(3):
        rt_a.append(ev(f"e{i}", f"fact {i}"))
    rt_b.sync()
    assert a.seen == b.seen == ["e0", "e1", "e2"]

    # The index is rebuilt with another threshold and both restart with it, each in a new process.
    db_a, db_b = reopen(), reopen()
    a2, b2 = ScanKeys(), ScanKeys()
    spy_a2, spy_b2 = SpyEmbedder(TxnSpy(db_a)), SpyEmbedder(TxnSpy(db_b))
    rebuild(db_a, spy_a2, threshold=0.5)
    rt_a = runtime(a2, db_a, spy_a2, threshold=0.5, name="a")
    rt_b = runtime(b2, db_b, spy_b2, threshold=0.5, name="b")
    assert rt_a.sync() == 3
    assert rt_b.sync() == 3

    assert a2.seen == b2.seen == ["e0", "e1", "e2"]
    for rt in (rt_a, rt_b):
        assert (rt.state.cursor(), rt.state.get("n")) == (3, 3)  # started over, not added to
        assert [k for k, _ in rt.state.scan()] == ["n"]
        assert rt.state.version() == composite(None, fingerprint(db_a, 0.5))
    for engine in (a, b, a2, b2):
        assert engine.keys == {"n"}
    for spy in (spy_a, spy_b, spy_a2, spy_b2):
        assert spy.inside == []


def test_an_engine_can_use_runtime_looking_keys():
    class UsesRuntimeKeys(ProbeEngine):
        def observe(self, event, ctx):
            if event.id == "e1":
                ctx.state.put("_runtime/x", 1)
                ctx.state.put("_runtime/gone", 2)
            else:
                self.read = ctx.state.get("_runtime/x")
                self.keys = [k for k, _ in ctx.state.scan("_runtime/")]
                ctx.state.delete("_runtime/gone")
            return super().observe(event, ctx)

    engine = UsesRuntimeKeys()
    rt = runtime(engine, InMemoryDatabase())
    rt.append(ev("e1"))
    rt.append(ev("e2"))
    assert engine.read == 1
    assert engine.keys == ["_runtime/gone", "_runtime/x"]
    assert dict(rt.state.scan()) == {"_runtime/x": 1, "n": 2}


@pytest.mark.parametrize("state_version", ["a@b", 'say "hi"', "one,two", "버전-✓", ""])
def test_composite_version_is_json_and_survives_odd_state_versions(state_version):
    class Versioned(ProbeEngine):
        pass

    Versioned.state_version = state_version
    db = InMemoryDatabase()
    engine = Versioned()
    rt = runtime(engine, db)
    rt.append(ev("e1"))
    rt.append(ev("e2"))

    stored = rt.state.version()
    assert json.loads(stored) == [state_version, fingerprint(db)]
    assert stored == composite(state_version, fingerprint(db))

    # The same version on the next sync and in a new process: no needless replay.
    engine.seen.clear()
    rt.sync()
    again = Versioned()
    runtime(again, db).sync()
    assert engine.seen == again.seen == []

    # Turning the index off replays once, and only once.
    off = Versioned()
    assert runtime(off, db, with_index=False).sync() == 2
    assert runtime(Versioned(), db, with_index=False).sync() == 0
    assert db.state("probe").version() == state_version

    # Turning it back on replays once, and only once.
    on = Versioned()
    assert runtime(on, db).sync() == 2
    assert runtime(Versioned(), db).sync() == 0
    assert json.loads(db.state("probe").version()) == [state_version, fingerprint(db)]


def test_an_engine_without_a_state_version_is_checked_when_an_index_is_present():
    db = InMemoryDatabase()
    rt = runtime(ProbeEngine(), db)
    rt.append(ev("e1"))
    assert json.loads(rt.state.version()) == [None, fingerprint(db)]

    # A changed fingerprint (here, the threshold of a rebuild) replays it.
    rebuild(db, threshold=0.5)
    replayed = ProbeEngine()
    assert runtime(replayed, db, threshold=0.5).sync() == 1
    assert json.loads(db.state("probe").version()) == [None, fingerprint(db, 0.5)]

    # Turning the index off replays it too: its state was built on an index it no longer reads, and
    # the knowledge base is recorded as owned under the indexed version.
    assert drop_index(db.vectors, db.transaction) is True
    unindexed = ProbeEngine()
    assert unindexed_runtime(unindexed, db).sync() == 1
    assert unindexed.seen == ["e1"] and db.state("probe").version() is None


def test_rebuild_by_another_process_with_the_same_config_replays_this_engine():
    db = InMemoryDatabase()
    engine = ProbeEngine()
    rt = runtime(engine, db)
    rt.append(ev("e1"))
    rt.append(ev("e2"))
    before = db.vectors.generation()

    # Another process finds the index reset (by a third config) and rebuilds it
    # with the same config as this runtime: same meta, a later generation.
    db.vectors.reset({"model": "someone-else", "group_threshold": 0.1, "store": "memory"})
    other = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
    assert other.rebuild(db.knowledge) is True
    assert db.vectors.meta() == rt.index.want() and db.vectors.generation() == before + 2

    # This runtime's own Index rebuilt nothing, yet its engine's state was built
    # on an older generation: it is replayed.
    engine.seen.clear()
    assert rt.sync() == 2
    assert engine.seen == ["e1", "e2"]
    assert rt.state.get("n") == 2
    assert json.loads(rt.state.version()) == [None, fingerprint(db)]
    assert rt.sync() == 0


def test_a_failed_rebuild_touches_neither_the_index_nor_engine_state():
    db = InMemoryDatabase()
    db.knowledge.put(item("k1", ("e1",)))
    rt = runtime(ProbeEngine(), db)
    rt.append(ev("e1"))
    rt.append(ev("e2"))
    # Another process promoted an item whose passage is not cached yet.
    db.knowledge.put(item("k2", ("e2",), body="never embedded"))
    state = db.state("probe")
    before = (state.cursor(), dict(state.scan()), state.version())
    index_before = (db.vectors.meta(), db.vectors.generation(), db.vectors.indexed())

    with pytest.raises(EmbeddingError):
        rebuild(db, FailingEmbedder({"deploy\nnever embedded"}), threshold=0.5)

    assert (state.cursor(), dict(state.scan()), state.version()) == before
    assert (db.vectors.meta(), db.vectors.generation(), db.vectors.indexed()) == index_before


class BumpsOnFirstVersionRead:
    """An EngineStateBackend that, on the first version() read, lets `hook`
    run first: another process changes the index between that read and the
    runtime's transaction."""

    def __init__(self, inner, hook):
        self._inner = inner
        self._hook = hook
        self.version_reads = 0
        self.resets = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def version(self):
        self.version_reads += 1
        if self.version_reads == 1:
            self._hook()
        return self._inner.version()

    def reset(self, version):
        self.resets.append(version)
        self._inner.reset(version)


def test_a_stale_version_check_inside_the_transaction_uses_the_fresh_fingerprint():
    db = InMemoryDatabase()
    db.events.append(ev("e1"))
    index = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
    index.rebuild(db.knowledge)
    seen_generation = db.vectors.generation()
    # Another process rebuilds with the same config after the first read.
    state = BumpsOnFirstVersionRead(db.state("probe"), lambda: db.vectors.reset(index.want()))
    engine = ProbeEngine()
    rt = Runtime(engine, db.events, db.knowledge, state, db.transaction, index=index)

    assert rt.sync() == 1

    assert db.vectors.generation() == seen_generation + 1
    assert state.resets == [composite(None, fingerprint(db))]  # one reset, with the fresh generation
    assert state.version() == composite(None, fingerprint(db))
    assert rt.sync() == 0 and engine.seen == ["e1"]  # no loop: it matches from now on


# ── replay sees what live delivery saw ──────────────────────────────────────


class Remembers(ProbeEngine):
    """Also records what search and neighbors return while observing."""

    state_version = "1"

    def __init__(self):
        super().__init__()
        self.found = {}

    def observe(self, event, ctx):
        self.found[event.id] = (
            [e.id for e, _ in ctx.similar.search("deploy uses tags", caller(), "org")],
            [e.id for e, _ in ctx.similar.neighbors(event.id, caller(), "org")],
        )
        return super().observe(event, ctx)


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_replay_sees_the_same_groups_and_neighbors_as_live_delivery(store, tmp_path):
    db = make_db(store, tmp_path)
    live = Remembers()
    rt = runtime(live, db)
    rt.append(ev("e1"))
    rt.append(ev("e2"))
    assert live.groups["e1"].members == ("e1",)
    assert live.groups["e2"].members == ("e1", "e2")
    assert live.found == {"e1": (["e1"], []), "e2": (["e1", "e2"], ["e1"])}

    class Bumped(Remembers):
        state_version = "2"

    replay = Bumped()
    assert runtime(replay, db).sync() == 2
    assert replay.groups == live.groups  # e1 -> ('e1',), not ('e1', 'e2')
    assert replay.found == live.found


def test_tick_and_handoff_read_up_to_the_engines_cursor():
    db = InMemoryDatabase()
    engine = ProbeEngine()
    rt = runtime(engine, db)
    rt.append(ev("e1"))
    # Indexed by another engine's runtime, not yet delivered to this one.
    other = runtime(ProbeEngine(), db, name="other")
    db.events.append(ev("e2"))
    other.sync()
    assert db.vectors.indexed() == 2 and rt.state.cursor() == 1

    staged = rt._staged()
    similar = rt._ctx(staged, rt.state.cursor()).similar
    assert similar.group("e2") is None
    assert [e.id for e, _ in similar.search("deploy uses tags", caller(), "personal")] == ["e1"]


# ── fix round 1: another config on the store, an outage ─────────────────────


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_two_runtimes_with_different_configs_do_not_mix_vectors(store, tmp_path):
    shared = InMemoryDatabase()

    def reopen():
        return shared if store == "memory" else SqliteDatabase(tmp_path / "core.db")

    db_a, db_b = reopen(), reopen()
    for i in range(3):
        db_a.events.append(ev(f"e{i}", f"fact number {i}"))
    ran = []

    def meanwhile(texts, kind):
        if not ran:
            ran.append(True)
            rebuild(db_b, HashEmbedder(dim=128))  # the index is rebuilt for another model while a embeds

    engine = ProbeEngine()
    a = runtime(engine, db_a, SpyEmbedder(TxnSpy(db_a), during=meanwhile), name="a")

    with pytest.raises(IndexMismatch):  # it terminates, and a does not take the index back
        a.sync()

    assert ran == [True]
    assert db_a.vectors.meta() == {**a.index.want(), "model": "hash-128"}
    assert list(db_a.vectors.vectors("event")) == []  # none of a's vectors in the other model's index
    assert engine.seen == []


def test_an_index_that_keeps_being_rebuilt_away_and_back_is_raised_after_three_retries():
    db = InMemoryDatabase()
    db.events.append(ev("e1"))
    rt = runtime(ProbeEngine(), db)
    rebuilds = []
    real = rt._check_versions

    def check_then_lose():
        # Rebuilt for this config when it is checked, and for another before the event is indexed.
        rebuilds.append(True)
        db.vectors.reset(rt.index.want())
        real()
        db.vectors.reset({"model": "someone-else", "group_threshold": 0.1, "store": "memory"})

    rt._check_versions = check_then_lose

    with pytest.raises(IndexConfigChanged):
        rt.sync()
    assert len(rebuilds) == 4  # the first check, then three retries


def test_an_outage_sync_makes_at_most_two_failing_calls():
    db = InMemoryDatabase()
    emb = SpyEmbedder(TxnSpy(db), fail_on={"t1"})
    engine = ProbeEngine()
    rt = runtime(engine, db, emb)
    rt.sync()
    for i in range(3):
        db.events.append(ev(f"e{i}", f"t{i}"))

    with pytest.raises(EmbeddingError):
        rt.sync()

    failing = [texts for texts in emb.batches if "t1" in texts]
    assert len(failing) <= 2, emb.batches
    assert engine.seen == ["e0"]  # the error belongs to e1: the event before it was delivered
    assert (rt.state.cursor(), db.vectors.indexed()) == (1, 1)
    emb.fail_on.clear()
    assert rt.sync() == 2


# ── delivery outside the write transaction ──────────────────────────────────


@pytest.mark.parametrize("store", STORES)
def test_observe_never_runs_inside_a_write_transaction(store, tmp_path, request):
    from spy import SpyEngine

    db = reopener(store, tmp_path, request)()
    spy = TxnSpy(db)
    for with_index in (True, False):
        engine = SpyEngine(spy, on_tick=[Promote(item("t1", ("e1",)))])
        index = Index(db.vectors, SpyEmbedder(spy), 0.9, transaction=spy.transaction) if with_index else None
        rt = Runtime(engine, db.events, db.knowledge, db.state(f"spy-{with_index}"), spy.transaction,
                     max_failures=3, index=index)
        if with_index:
            rt.append(ev("e1"))
            rt.append(ev("e2", "payments need vendor v2"))
        else:
            rt.sync()  # another engine on the same store, without an index
        rt.tick(T0)
        rt.handoff("deploy", caller(), "personal")

        assert [m for m, _ in engine.calls] == ["observe", "observe", "tick", "handoff"]
        assert engine.inside == []
        assert rt.state.cursor() == 2 and {"k-e1", "k-e2", "t1"} <= {k.id for k in db.knowledge.all()}


class Switchable(CountingEmbedder):
    """HashEmbedder that fails every call while `down`, or only queries when `queries_down`."""

    def __init__(self):
        super().__init__()
        self.down = False
        self.queries_down = False

    def embed(self, texts, kind):
        if self.down or (self.queries_down and kind == "query"):
            raise EmbeddingError("embedding backend is down")
        return super().embed(texts, kind)


@pytest.mark.parametrize("store", STORES)
def test_embedding_outage_does_not_block_appends(store, tmp_path, request):
    db = reopener(store, tmp_path, request)()
    emb = Switchable()
    engine = ProbeEngine(decisions={"e1": [Promote(item("k1", ("e1",)))]})
    rt = runtime(engine, db, emb, max_failures=1)
    rt.sync()
    emb.down = True

    with pytest.raises(EmbeddingError):
        rt.append(ev("e1"))
    with pytest.raises(EmbeddingError):  # another try is not counted toward max_failures
        rt.sync()

    assert [e.id for _, e in db.events.since(0)] == ["e1"]  # the record is stored
    assert (engine.seen, rt.state.cursor(), rt.pending(), rt.skipped, rt._failures) == ([], 0, 1, [], {})

    emb.down = False
    assert rt.sync() == 1
    assert rt.sync() == 0  # exactly once
    assert (engine.seen, rt.state.cursor(), rt.pending(), rt.skipped) == (["e1"], 1, 0, [])
    assert [k.id for k in db.knowledge.all()] == ["k1"]


def test_an_embedding_error_raised_inside_observe_is_not_counted_as_an_engine_failure():
    db = InMemoryDatabase()
    emb = Switchable()
    engine = ProbeEngine(ask_knowledge=True)  # observe embeds its event's text as a query
    rt = runtime(engine, db, emb, max_failures=1)
    emb.queries_down = True

    with pytest.raises(EmbeddingError):
        rt.append(ev("e1"))

    assert (rt.state.cursor(), rt.skipped, rt._failures) == (0, [], {})
    emb.queries_down = False
    assert rt.sync() == 1
    assert engine.seen == ["e1", "e1"]


def test_failures_stay_empty_after_an_index_config_change():
    db = InMemoryDatabase()
    db.events.append(ev("e1"))
    rt = runtime(ProbeEngine(), db, max_failures=1)
    real = rt._check_versions

    def check_then_lose():
        db.vectors.reset(rt.index.want())
        real()
        db.vectors.reset({"model": "someone-else", "group_threshold": 0.1, "store": "memory"})

    rt._check_versions = check_then_lose
    with pytest.raises(IndexConfigChanged):
        rt.sync()
    assert (rt._failures, rt.skipped, rt.state.cursor()) == ({}, [], 0)


class BlockingProbe(ProbeEngine):
    """ProbeEngine whose first observe of `block_on` signals `entered` and
    waits for `release`."""

    def __init__(self, block_on, **kw):
        super().__init__(**kw)
        self.block_on = block_on
        self.entered, self.release = threading.Event(), threading.Event()

    def observe(self, event, ctx):
        decisions = super().observe(event, ctx)
        if event.id == self.block_on and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(10), "never released"
        return decisions


@pytest.mark.parametrize("store", STORES)
def test_commit_is_refused_when_the_index_generation_changed_during_observe(store, tmp_path, request):
    open_db = reopener(store, tmp_path, request)
    db = open_db()
    engine = BlockingProbe("e1", decisions={"e1": [Promote(item("k1", ("e1",)))]})
    rt = runtime(engine, db)
    rt.sync()  # builds the index
    db.events.append(ev("e1"))
    generation = db.vectors.generation()

    pool = ThreadPoolExecutor(max_workers=1)
    worker = pool.submit(rt.sync)
    try:
        assert engine.entered.wait(10), "observe never started"
        other = open_db()  # the index is rebuilt with another threshold
        Index(other.vectors, HashEmbedder(), 0.5, transaction=other.transaction).rebuild(other.knowledge)
        assert other.vectors.generation() == generation + 1
    finally:
        engine.release.set()
        pool.shutdown(wait=True)

    with pytest.raises(IndexMismatch):
        worker.result()
    assert engine.seen == ["e1"]  # its result was refused, and it is not observed on the other index
    assert rt.state.cursor() == 0 and rt.state.get("n") is None
    assert db.vectors.meta()["group_threshold"] == 0.5 and db.vectors.generation() == generation + 1
    assert db.knowledge.all() == []


def test_group_threshold_one_and_one_point_zero_give_the_same_version():
    db = InMemoryDatabase()
    a = Index(db.vectors, HashEmbedder(), 1)
    b = Index(db.vectors, HashEmbedder(), 1.0)
    assert json.dumps(a.fingerprint()) == json.dumps(b.fingerprint())
    assert json.dumps(a.want()) == json.dumps(b.want())


@pytest.mark.parametrize("store", STORES)
def test_commit_is_refused_when_the_index_generation_moved_twice_during_observe(store, tmp_path, request):
    """During observe another config rebuilds (g+1), ours rebuilds back (g+2)
    and indexes past e1, and no runtime of this engine resets it: the index
    looks as it did, only its generation says it is another. The result
    built at g must not commit; the engine is replayed on g+2."""
    open_db = reopener(store, tmp_path, request)
    db = open_db()
    engine = BlockingProbe("e1")
    rt = runtime(engine, db)
    rt.sync()
    db.events.append(ev("e1"))
    g = db.vectors.generation()

    pool = ThreadPoolExecutor(max_workers=1)
    worker = pool.submit(rt.sync)
    try:
        assert engine.entered.wait(10), "observe never started"
        other = open_db()
        Index(other.vectors, HashEmbedder(), 0.5, transaction=other.transaction).rebuild(other.knowledge)
        ours = Index(other.vectors, HashEmbedder(), 0.9, transaction=other.transaction)
        ours.rebuild(other.knowledge)
        ours.advance(other.events, 1)
        assert (other.vectors.generation(), other.vectors.indexed()) == (g + 2, 1)
    finally:
        engine.release.set()
        pool.shutdown(wait=True)

    assert worker.result() == 1
    assert json.loads(rt.state.version())[1]["generation"] == g + 2  # built on the current index
    assert rt.state.cursor() == 1 and rt.state.get("n") == 1  # replayed from a reset, applied once
    assert len(engine.seen) >= 2


@pytest.mark.parametrize("store", STORES)
def test_commit_is_refused_when_the_index_fell_behind_during_observe(store, tmp_path, request):
    """The generation stays, but another process moves the index position
    back before e1 while it is observed: the commit is refused, e1 is
    indexed again and observed again, and the index ends at or past the
    engine's cursor."""
    open_db = reopener(store, tmp_path, request)
    db = open_db()
    engine = BlockingProbe("e1")
    rt = runtime(engine, db)
    rt.sync()
    db.events.append(ev("e1"))
    g = db.vectors.generation()

    pool = ThreadPoolExecutor(max_workers=1)
    worker = pool.submit(rt.sync)
    try:
        assert engine.entered.wait(10), "observe never started"
        other = open_db()
        with other.transaction():
            other.vectors.set_indexed(1, 0, g)  # held back, same generation
    finally:
        engine.release.set()
        pool.shutdown(wait=True)

    assert worker.result() == 1
    assert engine.seen == ["e1", "e1"]
    assert db.vectors.generation() == g
    assert db.vectors.indexed() >= rt.state.cursor() == 1
    assert rt.state.get("n") == 1


# ── knowledge vectors and rebuilds by other processes ───────────────────────


class RacingEmbedder(HashEmbedder):
    """HashEmbedder that runs `meanwhile` once, after its next embed call: another
    process acts between this one's embedding and its write."""

    def __init__(self, dim=256):
        super().__init__(dim)
        self.meanwhile = None

    def embed(self, texts, kind):
        out = super().embed(texts, kind)
        meanwhile, self.meanwhile = self.meanwhile, None
        if meanwhile is not None:
            meanwhile()
        return out


@pytest.mark.parametrize("store", STORES)
def test_a_knowledge_vector_never_lands_in_an_index_built_for_another_model(store, tmp_path, request):
    open_db = reopener(store, tmp_path, request)
    db_a, db_b = open_db(), open_db()
    emb = RacingEmbedder(256)
    a = Index(db_a.vectors, emb, 0.9, transaction=db_a.transaction)
    b = Index(db_b.vectors, HashEmbedder(64), 0.9, transaction=db_b.transaction)
    a.rebuild(db_a.knowledge)
    db_a.knowledge.put(item("k1", ("e1",), body="hello world"))
    emb.meanwhile = lambda: b.rebuild(db_b.knowledge)  # another model takes the index

    a.sync_knowledge(db_a.knowledge, ["k1"])

    assert db_b.vectors.meta() == b.want()
    assert len(db_b.vectors.vector("knowledge", "k1")) == 64  # B's own vector, not A's
    b.sync_knowledge(db_b.knowledge, None)
    hits = b.reader(db_b.events, db_b.knowledge).knowledge("hello", caller(), "personal")
    assert [k.id for k, _ in hits] == ["k1"]


@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_a_re_promoted_id_keeps_its_vector(store, tmp_path):
    """An id is demoted, then promoted again with the same text between the
    sync's read and its write: its vector is not deleted."""
    db = make_db(store, tmp_path)
    emb = RacingEmbedder(256)
    index = Index(db.vectors, emb, 0.9, transaction=db.transaction)
    index.rebuild(db.knowledge)
    k1 = item("k1", ("e1",), body="hello world")
    db.knowledge.put(k1)
    index.sync_knowledge(db.knowledge, ["k1"])
    db.knowledge.remove("k1")
    db.knowledge.put(item("k2", ("e1",), body="something new"))  # gives the sync something to embed
    emb.meanwhile = lambda: db.knowledge.put(k1)

    index.sync_knowledge(db.knowledge, ["k1", "k2"])

    assert db.vectors.vector("knowledge", "k1") is not None
    assert db.vectors.vector("knowledge", "k2") is not None


@pytest.mark.parametrize("store", STORES)
def test_two_rebuilds_back_to_back_cost_one_generation(store, tmp_path, request):
    open_db = reopener(store, tmp_path, request)
    db_a, db_b = open_db(), open_db()
    db_a.knowledge.put(item("k1", ("e1",), body="hello world"))
    emb = RacingEmbedder(256)
    a = Index(db_a.vectors, emb, 0.9, transaction=db_a.transaction)
    b = Index(db_b.vectors, HashEmbedder(256), 0.9, transaction=db_b.transaction)
    g = db_a.vectors.generation()
    emb.meanwhile = lambda: b.rebuild(db_b.knowledge)  # B rebuilds while A embeds

    assert a.rebuild(db_a.knowledge) is False  # B rebuilt it first

    assert db_a.vectors.generation() == g + 1


@pytest.mark.parametrize("store", STORES)
def test_knowledge_vectors_follow_the_reset_that_follows_our_own_rebuild(store, tmp_path, request):
    """The index is rebuilt for this Runtime's config (another config had it)
    after the Runtime had already reconciled once, and an item is put after
    the rebuild's snapshot. The rebuild resets the engine, which empties the
    knowledge base; the full compare that follows leaves no vector behind
    for any of it."""
    open_db = reopener(store, tmp_path, request)
    db = open_db()
    db.knowledge.put(item("k1", ("e1",), body="hello world"))
    rt = runtime(ProbeEngine(), db)
    rt.sync()  # reconciled once
    other = open_db()
    Index(other.vectors, HashEmbedder(64), 0.9, transaction=other.transaction).rebuild(other.knowledge)
    real = rt.index._knowledge_vectors

    def snapshot_then_promote(items):
        vectors = real(items)
        if not db.knowledge.get(["k2"]):
            open_db().knowledge.put(item("k2", ("e1",), body="promoted meanwhile"))
        return vectors

    rt.index._knowledge_vectors = snapshot_then_promote
    assert rt.index.rebuild(db.knowledge) is True
    rt.sync()

    assert db.vectors.meta() == rt.index.want()
    assert db.knowledge.all() == []
    assert list(db.vectors.vectors("knowledge")) == []


@pytest.mark.parametrize("store", STORES)
def test_a_tick_over_an_index_rebuilt_twice_is_dropped(store, tmp_path, request):
    """While the engine ticks, another config rebuilds the index and ours
    rebuilds it back: the cursor and epoch did not move, but the tick was
    made on another index generation, so it is dropped."""
    open_db = reopener(store, tmp_path, request)
    db = open_db()

    class Rebuilding(ProbeEngine):
        meanwhile = None

        def tick(self, now, ctx):
            if self.meanwhile is not None:
                meanwhile, self.meanwhile = self.meanwhile, None
                meanwhile()
            ctx.state.put("ticked", True)
            return [Promote(item("t1", ("e1",)))]

    engine = Rebuilding()
    rt = runtime(engine, db)
    rt.append(ev("e1"))
    other = open_db()

    def rebuild_twice():
        Index(other.vectors, HashEmbedder(), 0.5, transaction=other.transaction).rebuild(other.knowledge)
        Index(other.vectors, HashEmbedder(), 0.9, transaction=other.transaction).rebuild(other.knowledge)

    engine.meanwhile = rebuild_twice
    epoch = rt.state.epoch()
    rt.tick(T0)

    assert rt.state.get("ticked") is None and db.knowledge.get(["t1"]) == []
    assert rt.state.epoch() == epoch


# ── round trips to PostgreSQL ───────────────────────────────────────────────


class _CountingConnection:
    """Wraps a psycopg connection and counts the statements sent through it."""

    def __init__(self, inner):
        self._inner = inner
        self.statements = []

    def execute(self, query, params=None, **kw):
        self.statements.append(" ".join(str(query).split())[:80])
        return self._inner.execute(query, params, **kw)

    def cursor(self, *args, **kw):
        outer = self
        inner = self._inner.cursor(*args, **kw)

        class _Cursor:
            def __enter__(self):
                inner.__enter__()
                return self

            def __exit__(self, *exc):
                return inner.__exit__(*exc)

            def executemany(self, query, rows, **kw):
                outer.statements.append(" ".join(str(query).split())[:80])
                return inner.executemany(query, rows, **kw)

            def __getattr__(self, name):
                return getattr(inner, name)

        return _Cursor()

    def __getattr__(self, name):
        return getattr(self._inner, name)


# A record on PostgreSQL (append, index, deliver, commit) took 65 statements before
# the reads were combined. Each is a round trip, which is what a remote database costs.
ROUND_TRIPS_PER_RECORD = 49


def test_a_record_takes_few_round_trips_to_postgresql(request):
    from cortex_mem.store.postgres import PostgresDatabase

    db = PostgresDatabase(request.getfixturevalue("pg_schema"))
    request.addfinalizer(db.close)
    rt = runtime(ProbeEngine(), db)
    for i in range(5):
        rt.append(ev(f"w{i}", f"warm up record {i}"))
    counting = _CountingConnection(db.con)
    db.con = counting

    rt.append(ev("e1", "deploy uses release tags"))

    db.con = counting._inner
    assert len(counting.statements) <= ROUND_TRIPS_PER_RECORD, "\n".join(counting.statements)


@pytest.mark.parametrize("store", STORES)
def test_recall_answers_from_what_was_delivered_during_an_embedding_outage(store, tmp_path, request):
    db = reopener(store, tmp_path, request)()
    emb = Switchable()
    engine = ProbeEngine(handoffs=[Handoff("k1", "deploy uses tags", ("e1",))])
    rt = runtime(engine, db, emb)
    rt.append(ev("e1"))
    emb.down = True
    with pytest.raises(EmbeddingError):
        rt.append(ev("e2", "payments need vendor v2"))

    # The waiting event and the query cannot be embedded: recall answers without them, with the
    # similarity reader left out as when no index is configured, and still records the recall.
    handoffs = rt.recall("deploy", caller(), "personal", event_id="r1")

    assert [h.id for h in handoffs] == ["k1"]
    assert engine.similar["handoff"] is None
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2", "r1"]
    assert rt.waiting() == ["e2", "r1"]

    emb.down = False
    rt.sync()
    assert rt.waiting() == []


def test_a_query_that_cannot_be_embedded_is_answered_by_text_search():
    db = InMemoryDatabase()
    emb = Switchable()
    engine = ProbeEngine(handoffs=[Handoff("k1", "deploy uses tags", ("e1",))])
    rt = runtime(engine, db, emb)
    rt.append(ev("e1"))
    emb.queries_down = True  # passages embed, queries fail

    assert [h.id for h in rt.handoff("deploy", caller(), "personal")] == ["k1"]
    assert engine.similar["handoff"] is None  # asked again without the similarity reader
