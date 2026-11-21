"""Embedding behaves the same on every store: groups and search results agree
between the in-memory and SQLite stores, two runtimes on one file embed each
text once, and an engine that ignores ctx.similar behaves as without an index."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from cortex_mem.contract import Caller, Event
from cortex_mem.embed import HashEmbedder
from cortex_mem.engines.evidence import EvidenceEngine
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.base import CursorConflict
from cortex_mem.store.memory import InMemoryDatabase
from cortex_mem.store.sqlite import SqliteDatabase

T0 = datetime(2026, 11, 15, tzinfo=UTC)

STORES = ["memory", "sqlite", "pg"]
OPENED = []  # PostgreSQL connections opened by a test; closed before its schemas are dropped


class CountingEmbedder:
    def __init__(self):
        self.inner = HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.calls = []

    def embed(self, texts, kind):
        self.calls.extend((t, kind) for t in texts)
        return self.inner.embed(texts, kind)


@pytest.fixture
def where(tmp_path, request):
    """where(kind, name): a place for one store of that kind (a file, or a fresh PG schema URL).

    A PG kind skips the test when CORTEX_TEST_PG_URL is unset.
    """
    new_schema = None

    def close_all():
        while OPENED:
            OPENED.pop().close()

    def locate(kind, name):
        nonlocal new_schema
        if kind != "pg":
            return tmp_path / name
        if new_schema is None:
            new_schema = request.getfixturevalue("pg_schemas")
            request.addfinalizer(close_all)  # runs before the schemas are dropped
        return new_schema()

    return locate


def make_db(kind, path):
    if kind == "pg":
        from cortex_mem.store.postgres import PostgresDatabase

        db = PostgresDatabase(path)
        OPENED.append(db)
        return db
    return InMemoryDatabase() if kind == "memory" else SqliteDatabase(path)


def ev(id, text, minutes, kind="record", user="u1", team="t1", scope="personal", **kw):
    return Event(
        id=id, kind=kind, at=T0 + timedelta(minutes=minutes), user=user, team=team,
        session=f"s-{id}", scope=scope, text=text, **kw,
    )


def events():
    """Twenty events: personal of two users, team of two teams, org; near
    duplicates so groups form; an outcome, a recall and a redacted record that
    are not embedded."""
    rows = [
        ("p1", "deploy uses tags", {}),
        ("p2", "deploy uses tags", {}),
        ("p3", "deploy uses git tags to release", {}),
        ("p4", "database migrations run before deploy", {}),
        ("p5", "database migrations run before deploy", {}),
        ("q1", "deploy uses tags", {"user": "u2"}),
        ("q2", "payments need vendor v2", {"user": "u2"}),
        ("t1", "payments need vendor v2", {"scope": "team"}),
        ("t2", "payments need vendor v2 sandbox", {"scope": "team"}),
        ("t3", "deploy uses tags", {"scope": "team"}),
        ("t4", "cache keys include the tenant id", {"scope": "team", "user": "u2"}),
        ("t5", "cache keys include the tenant id", {"scope": "team", "user": "u2"}),
        ("x1", "payments need vendor v2", {"scope": "team", "user": "u9", "team": "t9"}),
        ("x2", "other team secret rollout plan", {"scope": "team", "user": "u9", "team": "t9"}),
        ("o1", "style guide prefers small functions", {"scope": "org"}),
        ("o2", "style guide prefers small functions", {"scope": "org", "user": "u9", "team": "t9"}),
        ("o3", "deploy uses tags", {"scope": "org"}),
        ("r1", "it worked", {"kind": "outcome", "outcome": "success", "refs": ("p1",)}),
        ("r2", "deploy tags", {"kind": "recall"}),
        ("r3", "", {"text_redacted": True}),
    ]
    return [ev(id, text, i, **kw) for i, (id, text, kw) in enumerate(rows)]


class Quiet:
    """An engine that does nothing: the index is what is under test."""

    def observe(self, event, ctx):
        return []

    def tick(self, now, ctx):
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return []


def runtime(db, embedder=None, name="probe"):
    index = Index(db.vectors, embedder or HashEmbedder(), 0.9, transaction=db.transaction)
    return Runtime(Quiet(), db.events, db.knowledge, db.state(name), db.transaction, index=index)


def callers():
    return [
        (Caller(user="u1", team="t1", session="a", at=T0), "personal"),
        (Caller(user="u2", team="t1", session="b", at=T0), "team"),
        (Caller(user="u9", team="t9", session="c", at=T0), "org"),
    ]


QUERIES = ["deploy tags", "payments vendor", "database migrations", "style guide", "nothing alike zzz"]


def observe(db, rt):
    reader = rt.index.reader(db.events, db.knowledge)
    ids = [e.id for e in events()]
    groups = {id: reader.group(id) for id in ids}
    searches = {
        (q, c.user, scope): [(e.id, s) for e, s in reader.search(q, c, scope, k=50)]
        for q in QUERIES
        for c, scope in callers()
    }
    neighbors = {id: [(e.id, s) for e, s in reader.neighbors(id, callers()[1][0], "team")] for id in ids}
    return groups, searches, neighbors


def flow(kind, path):
    db = make_db(kind, path)
    rt = runtime(db)
    for e in events():
        rt.append(e)
    return observe(db, rt)


def assert_nonvacuous(memory):
    assert any(hits for hits in memory[1].values())  # equality is not vacuous
    assert any(hits for hits in memory[2].values())
    assert any(g and len(g.members) > 1 for g in memory[0].values())


def test_stores_agree_on_groups_and_search(where):
    memory = flow("memory", None)
    sqlite = flow("sqlite", where("sqlite", "a.db"))

    assert_nonvacuous(memory)
    assert memory[0] == sqlite[0]
    # Scores compared exactly: both stores keep float32 and score with the same Python code.
    assert memory[1] == sqlite[1]
    assert memory[2] == sqlite[2]


def test_two_postgres_schemas_agree_exactly(where):
    a = flow("pg", where("pg", ""))
    b = flow("pg", where("pg", ""))

    assert_nonvacuous(a)
    assert a == b  # groups and scores, exactly


def close_enough(left, right):
    """Same keys, same ids in the same order, scores within 1e-6."""
    assert left.keys() == right.keys()
    for key in left:
        assert [id for id, _ in left[key]] == [id for id, _ in right[key]], key
        for (_, x), (_, y) in zip(left[key], right[key], strict=True):
            assert abs(x - y) < 1e-6, key


def test_postgres_agrees_with_memory_within_tolerance(where):
    memory = flow("memory", None)
    pg = flow("pg", where("pg", ""))

    assert_nonvacuous(memory)
    assert memory[0] == pg[0]  # the same groups
    close_enough(memory[1], pg[1])
    close_enough(memory[2], pg[2])


@pytest.mark.parametrize("kind", STORES)
def test_the_same_events_make_the_same_groups_twice(kind, where):
    first = flow(kind, where(kind, "a.db"))
    second = flow(kind, where(kind, "b.db"))

    assert first == second


@pytest.mark.parametrize("kind", STORES)
def test_groups_are_what_the_scenario_intends(kind, where):
    groups = flow(kind, where(kind, "a.db"))[0]

    assert groups["p1"].members == ("p1", "p2")
    assert groups["p2"].id == groups["p1"].id
    assert groups["p4"].members == ("p4", "p5")
    assert groups["t1"].members == ("t1",)  # same words, other (scope, owner)
    assert groups["x1"].members == ("x1",)
    assert groups["o1"].scope == "org" and groups["o1"].members == ("o1", "o2")  # an org has one owner
    for id in ("r1", "r2", "r3"):
        assert groups[id] is None  # never embedded


def embeddable(all_events):
    return [e for e in all_events if e.kind == "record" and not e.text_redacted and e.text]


def assert_indexed_once(db, all_events):
    """Every embeddable event has exactly one vector and one group slot."""
    want = {e.id for e in embeddable(all_events)}
    vectors = [id for id, _ in db.vectors.vectors("event")]
    assert len(vectors) == len(want) and set(vectors) == want
    assert db.vectors.indexed() == list(db.events.since(0))[-1][0]
    seen = []
    for id in sorted(want):
        gid = db.vectors.group_of(id)
        members = db.vectors.members(gid)
        assert len(members) == len(set(members)) and id in members
        if members[0] == id:
            seen.extend(members)
    assert sorted(seen) == sorted(want)  # every event is a member of exactly one group


def unique(all_events):
    """Same events, but every record has words of its own, so every embedding
    is a real call and a re-index could not hide behind the text cache."""
    return [
        e if not e.text or e.kind != "record" else replace(e, text=f"{e.text} number {e.id}")
        for e in all_events
    ]


class RecordingBackend:
    """Delegates to a VectorBackend and logs every vector and group write, so a
    re-index is visible even though the writes are upserts. ignore_cursor
    makes a broken indexer: it reads its position as 0 and moves the shared
    position without checking it."""

    def __init__(self, inner, log, ignore_cursor=False):
        self._inner = inner
        self.log = log
        self._ignore_cursor = ignore_cursor

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def put_vector(self, kind, id, vector, digest=None):
        self.log.append(("put_vector", kind, id))
        self._inner.put_vector(kind, id, vector, digest)

    def put_group(self, event_id, *args, **kw):
        self.log.append(("put_group", event_id))
        self._inner.put_group(event_id, *args, **kw)

    def indexed(self):
        return 0 if self._ignore_cursor else self._inner.indexed()

    def position(self):
        meta, generation, _ = self._inner.position()
        return meta, generation, self.indexed()

    def set_indexed(self, expected, seq, generation):
        """Logged as ("set_indexed", expected, seq, accepted), refused attempts too."""
        if self._ignore_cursor:
            current = self._inner.indexed()
            expected, seq, generation = current, max(seq, current), self._inner.generation()
        try:
            self._inner.set_indexed(expected, seq, generation)
        except CursorConflict:
            self.log.append(("set_indexed", expected, seq, False))
            raise
        self.log.append(("set_indexed", expected, seq, True))


def recording_runtime(db, embedder, log):
    backend = RecordingBackend(db.vectors, log)
    index = Index(backend, embedder, 0.9, transaction=db.transaction)
    return Runtime(Quiet(), db.events, db.knowledge, db.state("probe"), db.transaction, index=index)


def writes(log, kind):
    return [entry[-1] for entry in log if entry[0] == kind and (kind == "put_group" or entry[1] == "event")]


def assert_written_once(log, all_events, complete):
    """No event was written twice; when complete, every embeddable one was."""
    want = sorted(e.id for e in embeddable(all_events))
    for kind in ("put_vector", "put_group"):
        got = writes(log, kind)
        assert len(got) == len(set(got)), f"{kind} wrote an event twice: {sorted(got)}"
        assert set(got) <= set(want)
        if complete:
            assert sorted(got) == want


def assert_claims_disjoint(log):
    """Accepted set_indexed attempts claim disjoint position ranges, and every
    vector or group write follows an accepted claim of its event: a refused
    attempt never wrote."""
    claims = [(e[1], e[2]) for e in log if e[0] == "set_indexed" and e[3]]
    covered = [p for lo, hi in claims for p in range(lo + 1, hi + 1)]
    assert len(covered) == len(set(covered)), f"overlapping claims: {claims}"
    last = None
    for entry in log:
        if entry[0] == "set_indexed":
            last = entry
        elif entry[0] == "put_group" or (entry[0] == "put_vector" and entry[1] == "event"):
            assert last is not None and last[3], f"{entry} written after a refused claim {last}"


def test_the_write_log_check_fails_on_a_duplicate():
    duplicated = [("put_vector", "event", "p1"), ("put_group", "p1"), ("put_vector", "event", "p1")]
    with pytest.raises(AssertionError, match="wrote an event twice"):
        assert_written_once(duplicated, events(), complete=False)
    with pytest.raises(AssertionError, match="overlapping claims"):
        assert_claims_disjoint([("set_indexed", 0, 2, True), ("set_indexed", 1, 2, True)])
    with pytest.raises(AssertionError, match="refused claim"):
        assert_claims_disjoint([("set_indexed", 0, 1, False), ("put_group", "p1")])


# who appends event i: b falls several events behind between its turns, so it
# must catch up on events another runtime indexed, and each sync is checked
PLAN = "aabbaaaaabaaabbbabab"


def run_two_runtimes(path, all_events, embedder, log, kind="sqlite"):
    db_a, db_b = make_db(kind, path), make_db(kind, path)
    runtimes = {
        "a": recording_runtime(db_a, embedder, log),
        "b": recording_runtime(db_b, embedder, log),
    }
    for i, (who, e) in enumerate(zip(PLAN, all_events, strict=True)):
        runtimes[who].append(e)
        assert_written_once(log, all_events, complete=False)
        runtimes["a"].sync()
        assert_written_once(log, all_events, complete=False)
        if who == "b" or i in (8, 12, 15):  # otherwise b stays behind for a stretch
            runtimes["b"].sync()
            assert_written_once(log, all_events, complete=False)
    runtimes["a"].sync()
    runtimes["b"].sync()
    return db_a, db_b, runtimes


@pytest.mark.parametrize("kind", ["sqlite", "pg"])
@pytest.mark.parametrize("texts", ["repeated", "unique"])
def test_two_runtimes_on_one_store_index_each_event_once(texts, kind, where):
    embedder, log = CountingEmbedder(), []
    all_events = events() if texts == "repeated" else unique(events())

    db_a, db_b, runtimes = run_two_runtimes(where(kind, "shared.db"), all_events, embedder, log, kind=kind)

    assert_written_once(log, all_events, complete=True)  # the writes themselves, across both runtimes
    assert_claims_disjoint(log)
    passages = [t for t, kind in embedder.calls if kind == "passage"]
    assert sorted(passages) == sorted({e.text for e in embeddable(all_events)})
    if texts == "unique":
        assert len(passages) == len(embeddable(all_events))
    assert all(kind == "passage" for _, kind in embedder.calls)
    assert_indexed_once(db_a, all_events)
    assert_indexed_once(db_b, all_events)
    assert observe(db_a, runtimes["a"]) == observe(db_b, runtimes["b"])


@pytest.mark.parametrize("kind", ["sqlite", "pg"])
def test_the_write_log_catches_an_indexer_that_ignores_the_shared_cursor(kind, where):
    """Control for the test above. After two runtimes indexed everything, an
    indexer that reads its position as 0 and moves the shared position
    unchecked re-indexes every event, and the write log shows it. With the
    checked set_indexed (only the read is wrong) the same indexer gets
    CursorConflict before its first write, and the log shows nothing twice."""
    path = where(kind, "shared.db")
    log = []
    run_two_runtimes(path, events(), CountingEmbedder(), log, kind=kind)
    assert_written_once(log, events(), complete=True)
    db = make_db(kind, path)
    last = list(db.events.since(0))[-1][0]

    class ReadsZero(RecordingBackend):
        def indexed(self):
            return 0

        def position(self):
            meta, generation, _ = self._inner.position()
            return meta, generation, 0

    checked = Index(ReadsZero(db.vectors, log), CountingEmbedder(), 0.9, transaction=db.transaction)
    with pytest.raises(CursorConflict):
        checked.advance(db.events, last)
    assert log[-1] == ("set_indexed", 0, 1, False)  # refused, and nothing written after it
    assert_written_once(log, events(), complete=True)
    assert_claims_disjoint(log)

    rogue = Index(RecordingBackend(db.vectors, log, ignore_cursor=True), CountingEmbedder(), 0.9,
                  transaction=db.transaction)
    rogue.advance(db.events, last)
    with pytest.raises(AssertionError, match="wrote an event twice"):
        assert_written_once(log, events(), complete=True)
    got = writes(log, "put_group")
    assert len(got) == 2 * len(set(got)), "the log shows every event written twice"


def evidence_scenario(with_index):
    db = InMemoryDatabase()
    index = Index(db.vectors, HashEmbedder(), 0.9) if with_index else None
    rt = Runtime(EvidenceEngine(), db.events, db.knowledge, db.state("evidence"), db.transaction, index=index)

    def e(id, text="", minutes=0, kind="record", signal="self", outcome="unknown", refs=()):
        return Event(
            id=id, kind=kind, at=T0 + timedelta(minutes=minutes), user="u1", team="t1", session=f"s-{id}",
            scope="team", text=text, signal=signal, outcome=outcome, refs=refs,
        )

    rt.append(e("a", "deploy uses make release, run make release from the root", 0))
    rt.append(e("b", "deploy uses tags", 1))
    rt.append(e("c", "deploy uses tags", 2))
    rt.append(e("ci", "passed", 3, kind="outcome", signal="test", outcome="success", refs=("b",)))
    rt.append(e("bad", "payments need vendor v2", 4))
    rt.append(e("f", "failed", 5, kind="outcome", signal="test", outcome="failure", refs=("k-bad",)))
    rt.tick(T0 + timedelta(days=1))
    caller = Caller(user="u9", team="t1", session="ask", at=T0 + timedelta(days=2))
    handed = {q: [(h.id, h.content, h.sources) for h in rt.handoff(q, caller, "team")]
              for q in ("deploy tags", "deploy make release", "payments")}
    kb = sorted(
        (i.id, i.scope, i.owner, i.title, i.body, i.sources) for i in db.knowledge.all()
    )
    return kb, handed, rt.state.cursor()


def test_an_engine_that_ignores_similarity_behaves_the_same_with_an_index():
    plain = evidence_scenario(with_index=False)
    indexed = evidence_scenario(with_index=True)

    assert plain[0], "the scenario must produce knowledge"
    assert plain[1]["deploy tags"], "and hand something off"
    assert indexed == plain
