from datetime import datetime, timezone

import pytest

from cortex_mem.contract import Caller, Event, Knowledge
from cortex_mem.store.sqlite import SqliteDatabase
from cortex_mem.testing import (
    DatabaseContract,
    EngineStateBackendContract,
    EventStoreContract,
    KnowledgeStoreContract,
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

    assert db.con.execute("PRAGMA user_version").fetchone()[0] == 3
    [old] = db.events.get(["e1"])
    assert (old.text, old.text_redacted) == ("old", False)
    db.events.append(ev("e2", "new"))
    assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]
