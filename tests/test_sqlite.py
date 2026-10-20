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
