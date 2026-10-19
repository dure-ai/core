"""EpisodeStore conformance. A new store implementation (e.g. PostgreSQL)
passes the contract when it passes these tests: add its factory to STORES."""

from datetime import datetime, timezone

import pytest

from cortex_mem.contract import Event, QueryContext
from cortex_mem.store.memory import InMemoryEpisodeStore
from cortex_mem.store.sqlite import SqliteEpisodeStore

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)
STORES = [InMemoryEpisodeStore, SqliteEpisodeStore]


@pytest.fixture(params=STORES, ids=lambda f: f.__name__)
def store(request):
    return request.param()


def ev(id, text="", kind="record", user="u1", scope="personal"):
    return Event(id=id, kind=kind, at=T0, user=user, session="s1", scope=scope, text=text)


def ctx(user="u1", scope="personal"):
    return QueryContext(user=user, session="s2", scope=scope)


def test_append_returns_event_id(store):
    assert store.append(ev("e1", "a")) == "e1"


def test_duplicate_event_id_is_rejected(store):
    store.append(ev("e1", "a"))
    with pytest.raises(ValueError):
        store.append(ev("e1", "b"))


def test_get_keeps_requested_order_and_skips_unknown(store):
    store.append(ev("e1", "a"))
    store.append(ev("e2", "b"))

    assert [e.id for e in store.get(["e2", "missing", "e1"])] == ["e2", "e1"]


def test_get_round_trips_every_field(store):
    original = Event(
        id="e1", kind="outcome", at=T0, user="u1", session="s1", scope="team",
        text="ci passed", agent="claude-code", signal="test", outcome="success",
        conditions={"service": "a"}, refs=("h1", "h2"), repo="org/repo", rev="abc123",
    )
    store.append(original)

    assert store.get(["e1"]) == [original]


def test_search_ranks_better_match_first(store):
    store.append(ev("weak", "deploy"))
    store.append(ev("strong", "deploy uses release tags"))

    assert [e.id for e, _ in store.search("deploy release tags", ctx())] == ["strong", "weak"]


def test_search_scores_are_descending(store):
    for i, text in enumerate(["deploy", "deploy tags", "deploy release tags"]):
        store.append(ev(f"e{i}", text))

    scores = [s for _, s in store.search("deploy release tags", ctx())]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(store):
    for i in range(5):
        store.append(ev(f"e{i}", "deploy tags"))

    assert len(store.search("deploy tags", ctx(), k=2)) == 2


def test_search_filters_by_kind(store):
    store.append(ev("rec", "cloud sql proxy"))
    store.append(ev("rcl", "cloud sql proxy", kind="recall"))

    assert [e.id for e, _ in store.search("cloud sql", ctx(), kinds={"record"})] == ["rec"]


def test_personal_scope_hides_other_users_events(store):
    store.append(ev("mine", "deploy uses tags", user="u1", scope="team"))
    store.append(ev("theirs_team", "deploy uses tags", user="u2", scope="team"))
    store.append(ev("theirs_personal", "deploy uses tags", user="u2", scope="personal"))

    personal = {e.id for e, _ in store.search("deploy tags", ctx(scope="personal"))}
    team = {e.id for e, _ in store.search("deploy tags", ctx(scope="team"))}

    assert personal == {"mine"}
    assert team == {"mine", "theirs_team"}
