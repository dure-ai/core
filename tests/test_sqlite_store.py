from datetime import datetime, timezone

from cortex_mem.contract import Event, QueryContext
from cortex_mem.store.sqlite import SqliteEpisodeStore

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)


def ev(id, text):
    return Event(id=id, kind="record", at=T0, user="u1", session="s1", scope="personal", text=text)


def ctx():
    return QueryContext(user="u1", session="s2", scope="personal")


def test_korean_word_matches_with_attached_particle():
    store = SqliteEpisodeStore()
    store.append(ev("e1", "배포가 실패하면 프록시를 먼저 확인"))
    store.append(ev("e2", "레디스 캐시는 정상"))

    assert [e.id for e, _ in store.search("배포 프록시", ctx())] == ["e1"]


def test_events_persist_across_connections(tmp_path):
    path = tmp_path / "episodes.db"
    SqliteEpisodeStore(path).append(ev("e1", "deploy uses tags"))

    reopened = SqliteEpisodeStore(path)

    assert [e.id for e in reopened.get(["e1"])] == ["e1"]
    assert [e.id for e, _ in reopened.search("deploy", ctx())] == ["e1"]


def test_events_replays_in_append_order():
    store = SqliteEpisodeStore()
    for id in ["b", "a", "c"]:
        store.append(ev(id, "x"))

    assert [e.id for e in store.events()] == ["b", "a", "c"]
