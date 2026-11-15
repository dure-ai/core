from datetime import datetime, timezone

import pytest

from cortex_mem.contract import Caller, Event, Knowledge
from cortex_mem.store.sqlite import SqliteDatabase
from cortex_mem.testing import (
    DatabaseContract,
    EngineStateBackendContract,
    EventStoreContract,
    KnowledgeStoreContract,
    VectorBackendContract,
)

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)


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
        return SqliteDatabase().vectors


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

    assert db.con.execute("PRAGMA user_version").fetchone()[0] == 4
    [old] = db.events.get(["e1"])
    assert (old.text, old.text_redacted) == ("old", False)
    db.events.append(ev("e2", "new"))
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]


def test_schema_3_file_migrates_to_4_and_keeps_events(tmp_path):
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

    assert db.query("PRAGMA user_version")[0][0] == 4
    assert [e.id for e in db.events.get(["e1"])] == ["e1"]
    db.vectors.put_vector("event", "e1", [1.0])
    assert db.vectors.vector("event", "e1") == [1.0]


def test_vectors_groups_and_cursor_survive_reopening(tmp_path):
    path = tmp_path / "v.db"
    db = SqliteDatabase(path)
    v = db.vectors
    v.reset({"model": "m", "group_threshold": 0.5})
    v.put_vector("event", "e1", [0.1, 0.2])
    v.put_vector("knowledge", "k1", [0.3])
    v.put_group("e1", "g1", "personal", "u1", 1)
    v.set_indexed(7)
    v.cache("m", "d", [0.5, 1 / 3])
    db.con.close()

    v = SqliteDatabase(path).vectors

    assert v.meta() == {"model": "m", "group_threshold": 0.5}
    assert v.vector("event", "e1") == [0.1, 0.2]
    assert v.vector("knowledge", "k1") == [0.3]
    assert v.group_of("e1") == "g1"
    assert v.partition("personal", "u1") == ["e1"]
    assert v.indexed() == 7
    assert v.cached("m", "d") == [0.5, 1 / 3]


def test_writes_roll_back_with_the_transaction():
    db = SqliteDatabase()
    with pytest.raises(RuntimeError), db.transaction():
        db.vectors.put_vector("event", "e1", [1.0])
        db.vectors.put_group("e1", "g1", "personal", "u1", 1)
        db.vectors.set_indexed(3)
        raise RuntimeError("boom")

    assert db.vectors.vector("event", "e1") is None
    assert db.vectors.group_of("e1") is None
    assert db.vectors.indexed() == 0


def test_vectors_is_the_same_object_on_each_access():
    db = SqliteDatabase()
    assert db.vectors is db.vectors
