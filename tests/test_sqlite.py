from datetime import UTC, datetime

import pytest

from cortex_mem.contract import Caller, Event, Knowledge
from cortex_mem.store.base import CursorConflict
from cortex_mem.store.sqlite import SqliteDatabase
from cortex_mem.testing import (
    DatabaseContract,
    EngineStateBackendContract,
    EventStoreContract,
    KnowledgeStoreContract,
    VectorBackendContract,
    f32,
)

T0 = datetime(2026, 10, 19, tzinfo=UTC)


class TestSqliteEventStore(EventStoreContract):
    def make(self):
        return SqliteDatabase().events


class TestSqliteKnowledgeBase(KnowledgeStoreContract):
    def make(self):
        return SqliteDatabase().knowledge


class TestSqliteStateBackend(EngineStateBackendContract):
    def make(self):
        return SqliteDatabase().state("engine")


class TestSqliteVectorBackend(VectorBackendContract):
    def make(self):
        return SqliteDatabase()


class TestSqliteDatabase(DatabaseContract):
    """Each open is a new connection to the same file, like a new server process."""

    def make(self, tmp_path):
        path = tmp_path / "cortex.db"
        return lambda: SqliteDatabase(path)


def ev(id, text, user="u1", scope="personal"):
    return Event(id=id, kind="record", at=T0, user=user, team="t1", session="s1", scope=scope, text=text)


def caller():
    return Caller(user="u1", team="t1", session="s2", at=T0)


def test_korean_word_matches_with_attached_particle():
    db = SqliteDatabase()
    db.events.append(ev("e1", "배포가 실패하면 프록시를 먼저 확인"))
    db.events.append(ev("e2", "레디스 캐시는 정상"))

    assert [e.id for e, _ in db.events.search("배포 프록시", caller(), "personal")] == ["e1"]


def test_korean_knowledge_search_uses_bigrams():
    db = SqliteDatabase()
    db.knowledge.put(Knowledge(id="k1", kind="knowledge", title="배포", body="배포가 실패하면 프록시 확인",
                               scope="personal", owner="u1", sources=("e1",)))

    assert [k.id for k, _ in db.knowledge.search("배포 실패", caller(), "personal")] == ["k1"]


def test_replaced_knowledge_is_searched_by_its_new_text():
    db = SqliteDatabase()
    item = Knowledge(id="k1", kind="knowledge", title="t", body="redis cache", scope="personal",
                     owner="u1", sources=("e1",))
    db.knowledge.put(item)
    db.knowledge.put(Knowledge(**{**item.__dict__, "body": "postgres pool"}))

    assert db.knowledge.search("redis", caller(), "personal") == []
    assert [k.id for k, _ in db.knowledge.search("postgres", caller(), "personal")] == ["k1"]


def test_old_schema_is_refused(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, id)")
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="schema"):
        SqliteDatabase(path)


def test_schema_2_file_gains_text_redacted(tmp_path):
    import sqlite3

    path = tmp_path / "v2.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE events (seq INTEGER PRIMARY KEY, id, kind, at, user, team, session, scope, text, agent, "
        "signal, outcome, conditions, refs, repo, rev, UNIQUE (id))"
    )
    con.execute(
        "INSERT INTO events (id, kind, at, user, team, session, scope, text, agent, signal, outcome, conditions, "
        "refs, repo, rev) VALUES ('e1', 'record', ?, 'u1', 't1', 's1', 'personal', 'old', NULL, NULL, NULL, "
        "'{}', '[]', NULL, NULL)",
        (T0.isoformat(),),
    )
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()

    db = SqliteDatabase(path)

    assert db.con.execute("PRAGMA user_version").fetchone()[0] == 5
    [old] = db.events.get(["e1"])
    assert (old.text, old.text_redacted) == ("old", False)
    db.events.append(ev("e2", "new"))
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]


def test_schema_3_file_migrates_to_5_and_keeps_events(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    cols = (
        "id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
        "conditions, refs, repo, rev, text_redacted"
    )
    con.execute(f"CREATE TABLE events (seq INTEGER PRIMARY KEY, {cols}, UNIQUE (id))")
    con.execute("CREATE VIRTUAL TABLE events_fts USING fts5(words, grams)")
    con.execute("CREATE TABLE knowledge (id TEXT PRIMARY KEY, kind, title, body, scope, owner, sources, conditions)")
    con.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(words, grams)")
    con.execute("CREATE TABLE engine_state (engine TEXT, key TEXT, value TEXT, PRIMARY KEY (engine, key))")
    con.execute("CREATE TABLE engine_cursor (engine TEXT PRIMARY KEY, cursor INTEGER, version TEXT)")
    con.execute(
        "INSERT INTO events (id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
        "conditions, refs, repo, rev, text_redacted) VALUES "
        "('e1', 'record', ?, 'u1', 't1', 's1', 'personal', 'hello', NULL, NULL, NULL, '{}', '[]', NULL, NULL, 0)",
        (T0.isoformat(),),
    )
    con.execute("PRAGMA user_version = 3")
    con.commit()
    con.close()

    db = SqliteDatabase(path)

    assert db.query("PRAGMA user_version")[0][0] == 5
    assert [e.id for e in db.events.get(["e1"])] == ["e1"]
    db.vectors.put_vector("event", "e1", [1.0])
    assert db.vectors.vector("event", "e1") == [1.0]


def test_schema_4_file_drops_its_float64_vectors_and_cache_and_reembeds(tmp_path):
    """Schema 4 kept float64 BLOBs; read as float32 they are 2n garbage values.
    Opening such a file empties every vector table, the cache included, so the
    rebuild embeds again instead of reusing the old cache."""
    import sqlite3
    from array import array

    from cortex_mem.embed import HashEmbedder, text_digest
    from cortex_mem.index import Index
    from cortex_mem.runtime import Runtime

    path = tmp_path / "v4.db"
    SqliteDatabase(path).con.close()  # the tables are the same in 4 and 5; only the BLOB format changed
    text = "deploy uses tags"
    old = HashEmbedder().embed([text], "passage")[0]
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO events (seq, id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
        "conditions, refs, repo, rev, text_redacted) VALUES (1, 'e1', 'record', ?, 'u1', 't1', 's1', "
        "'personal', ?, NULL, NULL, NULL, '{}', '[]', NULL, NULL, 0)",
        (T0.isoformat(), text),
    )
    con.execute("INSERT INTO events_fts (rowid, words, grams) VALUES (1, 'deploy uses tags', '')")
    blob = array("d", old).tobytes()
    con.execute("INSERT INTO embed_cache VALUES ('hash-256', ?, ?)", (text_digest(text), blob))
    con.execute("INSERT INTO vectors (kind, id, seq, vec) VALUES ('event', 'e1', 1, ?)", (blob,))
    con.execute("INSERT INTO vector_groups VALUES ('e1', 'e1', 'personal', 'u1', 1)")
    con.execute("""INSERT INTO vector_meta VALUES ('meta', '{"model": "hash-256", "group_threshold": 0.9}')""")
    con.execute("INSERT INTO vector_meta VALUES ('indexed', '1')")
    con.execute("PRAGMA user_version = 4")
    con.commit()
    con.close()

    db = SqliteDatabase(path)

    assert db.query("PRAGMA user_version")[0][0] == 5
    for table in ("embed_cache", "vectors", "vector_groups", "vector_meta"):
        assert db.query(f"SELECT count(*) FROM {table}") == [(0,)], table
    assert [e.id for e in db.events.get(["e1"])] == ["e1"]  # events stay

    class Counting:
        model, dim = "hash-256", 256

        def __init__(self):
            self.calls = []

        def embed(self, texts, kind):
            self.calls.extend(texts)
            return HashEmbedder().embed(texts, kind)

    class Quiet:
        def observe(self, event, ctx):
            return []

        def tick(self, now, ctx):
            return []

        def handoff(self, query, caller, scope, ctx, k=6):
            return []

    emb = Counting()
    index = Index(db.vectors, emb, 0.9, transaction=db.transaction)
    rt = Runtime(Quiet(), db.events, db.knowledge, db.state("probe"), db.transaction, index=index)
    rt.sync()

    assert emb.calls == [text]  # embedded again, not read from the float64 cache
    assert db.vectors.vector("event", "e1") == f32(old)
    assert index.reader(db.events, db.knowledge).group("e1").members == ("e1",)
    db.con.close()
    assert SqliteDatabase(path).vectors.vector("event", "e1") == f32(old)  # a later open keeps 5's rows


def test_vectors_groups_and_cursor_survive_reopening(tmp_path):
    path = tmp_path / "v.db"
    db = SqliteDatabase(path)
    v = db.vectors
    v.reset({"model": "m", "group_threshold": 0.5})
    v.put_vector("event", "e1", [0.1, 0.2])
    v.put_vector("knowledge", "k1", [0.3], digest="dk")
    v.put_group("e1", "g1", "personal", "u1", 1)
    v.set_indexed(0, 7, v.generation())
    v.cache("m", "d", [0.5, 1 / 3])
    db.con.close()

    v = SqliteDatabase(path).vectors

    assert v.meta() == {"model": "m", "group_threshold": 0.5}
    assert v.vector("event", "e1") == f32([0.1, 0.2])
    assert v.vector("knowledge", "k1") == f32([0.3])
    assert v.group_of("e1") == "g1"
    assert v.partition("personal", "u1") == ["e1"]
    assert v.indexed() == 7
    assert v.generation() == 1
    assert v.knowledge_digests() == {"k1": "dk"}
    assert v.cached("m", "d") == f32([0.5, 1 / 3])


def test_generation_survives_reopening(tmp_path):
    path = tmp_path / "g.db"
    db = SqliteDatabase(path)
    assert db.vectors.generation() == 0
    db.vectors.reset({"model": "m", "group_threshold": 0.5})
    db.vectors.reset({"model": "m", "group_threshold": 0.5})
    db.con.close()

    db = SqliteDatabase(path)
    assert db.vectors.generation() == 2
    assert db.vectors.meta() == {"model": "m", "group_threshold": 0.5}
    with pytest.raises(CursorConflict):
        db.vectors.set_indexed(0, 1, 1)
    db.vectors.set_indexed(0, 1, 2)
    assert SqliteDatabase(path).vectors.indexed() == 1


def test_writes_roll_back_with_the_transaction():
    db = SqliteDatabase()
    with pytest.raises(RuntimeError), db.transaction():
        db.vectors.put_vector("event", "e1", [1.0])
        db.vectors.put_group("e1", "g1", "personal", "u1", 1)
        db.vectors.set_indexed(0, 3, db.vectors.generation())
        raise RuntimeError("boom")

    assert db.vectors.vector("event", "e1") is None
    assert db.vectors.group_of("e1") is None
    assert db.vectors.indexed() == 0


def test_vectors_is_the_same_object_on_each_access():
    db = SqliteDatabase()
    assert db.vectors is db.vectors


def _schema_3_file(path):
    """A file as main (schema 3) leaves it: no epoch, no vector tables."""
    import sqlite3

    con = sqlite3.connect(path)
    cols = (
        "id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
        "conditions, refs, repo, rev, text_redacted"
    )
    con.execute(f"CREATE TABLE events (seq INTEGER PRIMARY KEY, {cols}, UNIQUE (id))")
    con.execute("CREATE VIRTUAL TABLE events_fts USING fts5(words, grams)")
    con.execute("CREATE TABLE knowledge (id TEXT PRIMARY KEY, kind, title, body, scope, owner, sources, conditions)")
    con.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(words, grams)")
    con.execute("CREATE TABLE engine_state (engine TEXT, key TEXT, value TEXT, PRIMARY KEY (engine, key))")
    con.execute("CREATE TABLE engine_cursor (engine TEXT PRIMARY KEY, cursor INTEGER, version TEXT)")
    for seq, id in ((1, "e1"), (2, "e2")):
        con.execute(
            f"INSERT INTO events (seq, {cols}) VALUES (?, ?, 'record', ?, 'u1', 't1', 's1', 'personal', 'hello', "
            "NULL, NULL, NULL, '{}', '[]', NULL, NULL, 0)",
            (seq, id, T0.isoformat()),
        )
    con.execute("INSERT INTO engine_state VALUES ('promote_all', 'seen', '1')")
    con.execute("INSERT INTO engine_cursor VALUES ('promote_all', 1, NULL)")  # e1 delivered on main
    con.execute("PRAGMA user_version = 3")
    con.commit()
    con.close()


def test_a_schema_3_file_from_main_opens_and_delivers(tmp_path):
    from cortex_mem.engines.baseline import PromoteAllEngine
    from cortex_mem.runtime import Runtime

    path = tmp_path / "main.db"
    _schema_3_file(path)

    for _ in range(2):  # opening twice is fine
        db = SqliteDatabase(path)
        state = db.state("promote_all")
        assert (state.cursor(), state.get("seen")) == (1, 1)  # kept
        db.con.close()

    db = SqliteDatabase(path)
    rt = Runtime(PromoteAllEngine(), db.events, db.knowledge, db.state("promote_all"), db.transaction)
    assert rt.sync() == 1  # e2 only: e1 was delivered before
    assert rt.state.cursor() == 2 and rt.state.get("seen") == 1
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]


def test_a_schema_5_file_without_knowledge_digests_opens(tmp_path):
    """Schema 5 files from feat/core-embedding have vectors without a digest column."""
    import sqlite3

    path = tmp_path / "v5.db"
    SqliteDatabase(path).con.close()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE vectors")
    con.execute("CREATE TABLE vectors (kind TEXT, id TEXT, seq INTEGER, vec BLOB, PRIMARY KEY (kind, id))")
    con.commit()
    con.close()
    db = SqliteDatabase(path)
    db.vectors.put_vector("event", "e1", [1.0])
    db.con.close()

    for _ in range(2):
        db = SqliteDatabase(path)
        assert db.vectors.vector("event", "e1") == [1.0]  # kept
        db.vectors.put_vector("knowledge", "k1", [0.5], digest="d1")
        assert db.vectors.knowledge_digests() == {"k1": "d1"}
        db.con.close()


def test_with_block_closes_the_connection():
    import sqlite3

    with SqliteDatabase() as db:
        db.events.append(ev("e1", "x"))
    with pytest.raises(sqlite3.ProgrammingError):
        db.con.execute("SELECT 1")


def test_old_schema_error_closes_the_connection(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, id)")
    con.commit()
    con.close()
    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: opened.append(real_connect(*a, **kw)) or opened[-1])

    with pytest.raises(RuntimeError, match="schema"):
        SqliteDatabase(path)
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
