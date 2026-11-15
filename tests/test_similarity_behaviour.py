"""Embedding behaves the same on every store: groups and search results agree
between the in-memory and SQLite stores, two runtimes on one file embed each
text once, and an engine that ignores ctx.similar behaves as without an index."""

from datetime import UTC, datetime, timedelta

import pytest

from cortex_mem.contract import Caller, Event
from cortex_mem.embed import HashEmbedder
from cortex_mem.engines.evidence import EvidenceEngine
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase
from cortex_mem.store.sqlite import SqliteDatabase

T0 = datetime(2026, 11, 15, tzinfo=UTC)

STORES = ["memory", "sqlite"]


class CountingEmbedder:
    def __init__(self):
        self.inner = HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.calls = []

    def embed(self, texts, kind):
        self.calls.extend((t, kind) for t in texts)
        return self.inner.embed(texts, kind)


def make_db(kind, path):
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
    index = Index(db.vectors, embedder or HashEmbedder(), 0.9)
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


def test_stores_agree_on_groups_and_search(tmp_path):
    memory = flow("memory", None)
    sqlite = flow("sqlite", tmp_path / "a.db")

    assert memory[0] == sqlite[0]
    assert memory[1] == sqlite[1]  # scores compared exactly: both stores keep float64
    assert memory[2] == sqlite[2]


@pytest.mark.parametrize("kind", STORES)
def test_the_same_events_make_the_same_groups_twice(kind, tmp_path):
    first = flow(kind, tmp_path / "a.db")
    second = flow(kind, tmp_path / "b.db")

    assert first == second


@pytest.mark.parametrize("kind", STORES)
def test_groups_are_what_the_scenario_intends(kind, tmp_path):
    groups = flow(kind, tmp_path / "a.db")[0]

    assert groups["p1"].members == ("p1", "p2")
    assert groups["p2"].id == groups["p1"].id
    assert groups["p4"].members == ("p4", "p5")
    assert groups["t1"].members == ("t1",)  # same words, other (scope, owner)
    assert groups["x1"].members == ("x1",)
    assert groups["o1"].scope == "org" and groups["o1"].members == ("o1", "o2")  # an org has one owner
    for id in ("r1", "r2", "r3"):
        assert groups[id] is None  # never embedded


def test_two_runtimes_on_one_file_embed_each_text_once(tmp_path):
    path = tmp_path / "shared.db"
    embedder = CountingEmbedder()
    db_a, db_b = SqliteDatabase(path), SqliteDatabase(path)
    rt_a = runtime(db_a, embedder)
    rt_b = runtime(db_b, embedder)

    all_events = events()
    for i, e in enumerate(all_events):
        (rt_a if i % 2 == 0 else rt_b).append(e)
    rt_a.sync()
    rt_b.sync()

    embedded = [e.text for e in all_events if e.kind == "record" and not e.text_redacted and e.text]
    passages = [t for t, kind in embedder.calls if kind == "passage"]
    # Each distinct text once (the text cache covers repeats), and nothing else.
    assert sorted(passages) == sorted(set(embedded))
    assert all(kind == "passage" for _, kind in embedder.calls)

    # Both runtimes read the same index afterwards.
    assert observe(db_a, rt_a) == observe(db_b, rt_b)


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
