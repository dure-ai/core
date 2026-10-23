"""Conformance suites for store implementations. A store passes the contract
when a subclass of the matching suite passes:

    class TestSqliteEvents(EventStoreContract):
        def make(self):
            return SqliteDatabase(":memory:").events

DatabaseContract runs the runtime over a whole Database, so delivery
(exactly once, all or nothing, state versions) is checked per backend too.

Needs pytest; import it from tests only.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from .contract import Caller, Event, Knowledge, Promote
from .runtime import Runtime
from .store.base import CursorConflict

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)


def ev(id, text="", kind="record", user="u1", team="t1", scope="personal"):
    return Event(id=id, kind=kind, at=T0, user=user, team=team, session="s1", scope=scope, text=text)


def caller(user="u1", team="t1"):
    return Caller(user=user, team=team, session="s2", at=T0)


def item(id, body="deploy uses tags", scope="team", owner="t1"):
    return Knowledge(id=id, kind="knowledge", title=id, body=body, scope=scope, owner=owner, sources=("e1",))


class _Contract:
    def make(self) -> Any:
        raise NotImplementedError


class EventStoreContract(_Contract):
    @pytest.fixture
    def store(self):
        return self.make()

    def test_append_returns_event_id(self, store):
        assert store.append(ev("e1", "a")) == "e1"

    def test_duplicate_event_id_is_rejected(self, store):
        store.append(ev("e1", "a"))
        with pytest.raises(ValueError):
            store.append(ev("e1", "b"))

    def test_get_keeps_requested_order_and_skips_unknown(self, store):
        store.append(ev("e1", "a"))
        store.append(ev("e2", "b"))

        assert [e.id for e in store.get(["e2", "missing", "e1"])] == ["e2", "e1"]

    def test_get_round_trips_every_field(self, store):
        original = Event(
            id="e1", kind="outcome", at=T0, user="u1", team="t1", session="s1", scope="team",
            text="ci passed", agent="claude-code", signal="test", outcome="success",
            conditions={"service": "a"}, refs=("h1", "h2"), repo="org/repo", rev="abc123",
        )
        store.append(original)

        assert store.get(["e1"]) == [original]

    def test_since_returns_append_order_with_growing_positions(self, store):
        for id in ["b", "a", "c"]:
            store.append(ev(id, "x"))

        everything = list(store.since(0))
        assert [e.id for _, e in everything] == ["b", "a", "c"]
        positions = [seq for seq, _ in everything]
        assert positions == sorted(positions) and positions[0] >= 1
        assert [e.id for _, e in store.since(positions[1])] == ["c"]

    def test_search_ranks_better_match_first(self, store):
        store.append(ev("weak", "deploy"))
        store.append(ev("strong", "deploy uses release tags"))

        assert [e.id for e, _ in store.search("deploy release tags", caller(), "personal")] == ["strong", "weak"]

    def test_search_scores_are_descending(self, store):
        for i, text in enumerate(["deploy", "deploy tags", "deploy release tags"]):
            store.append(ev(f"e{i}", text))

        scores = [s for _, s in store.search("deploy release tags", caller(), "personal")]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_k(self, store):
        for i in range(5):
            store.append(ev(f"e{i}", "deploy tags"))

        assert len(store.search("deploy tags", caller(), "personal", k=2)) == 2

    def test_search_filters_by_kind(self, store):
        store.append(ev("rec", "cloud sql proxy"))
        store.append(ev("rcl", "cloud sql proxy", kind="recall"))

        assert [e.id for e, _ in store.search("cloud sql", caller(), "personal", kinds={"record"})] == ["rec"]

    def test_search_scope_widens_personal_team_org(self, store):
        store.append(ev("mine", "deploy uses tags", user="u1", scope="personal"))
        store.append(ev("team_mate", "deploy uses tags", user="u2", scope="team"))
        store.append(ev("team_mate_personal", "deploy uses tags", user="u2", scope="personal"))
        store.append(ev("other_team", "deploy uses tags", user="u3", team="t2", scope="team"))
        store.append(ev("org_wide", "deploy uses tags", user="u3", team="t2", scope="org"))

        def ids(scope):
            return {e.id for e, _ in store.search("deploy tags", caller(), scope)}

        assert ids("personal") == {"mine"}
        assert ids("team") == {"mine", "team_mate"}
        assert ids("org") == {"mine", "team_mate", "org_wide"}

class KnowledgeStoreContract(_Contract):
    @pytest.fixture
    def kb(self):
        return self.make()

    def test_knowledge_put_replaces_same_id(self, kb):
        kb.put(item("k1", "old"))
        kb.put(item("k1", "new"))

        assert [i.body for i in kb.get(["k1"])] == ["new"]

    def test_knowledge_remove_and_unknown_ids(self, kb):
        kb.put(item("k1"))
        kb.remove("k1")
        kb.remove("missing")

        assert kb.get(["k1"]) == []
        assert kb.all() == []

    def test_knowledge_all_is_ordered_by_id(self, kb):
        for id in ["b", "c", "a"]:
            kb.put(item(id))

        assert [i.id for i in kb.all()] == ["a", "b", "c"]

    def test_knowledge_round_trips_every_field(self, kb):
        original = Knowledge(
            id="deploy/tags", kind="skill", title="Deploy", body="1. tag\n2. push", scope="personal",
            owner="u1", sources=("e1", "e2"), conditions={"service": "api"},
        )
        kb.put(original)

        assert kb.get(["deploy/tags"]) == [original]

    def test_knowledge_search_respects_scope(self, kb):
        kb.put(item("mine", scope="personal", owner="u1"))
        kb.put(item("someone_else", scope="personal", owner="u2"))
        kb.put(item("my_team", scope="team", owner="t1"))
        kb.put(item("other_team", scope="team", owner="t2"))
        kb.put(item("org_wide", scope="org", owner="*"))

        def ids(scope):
            return {i.id for i, _ in kb.search("deploy tags", caller(), scope)}

        assert ids("personal") == {"mine"}
        assert ids("team") == {"mine", "my_team"}
        assert ids("org") == {"mine", "my_team", "org_wide"}

class EngineStateBackendContract(_Contract):
    @pytest.fixture
    def state(self):
        return self.make()

    def test_state_commit_applies_writes_deletes_and_cursor(self, state):
        state.commit(0, 3, {"a": 1, "b": {"x": [1, 2]}}, set())
        state.commit(3, 5, {"c": "z"}, {"a"})

        assert state.cursor() == 5
        assert state.get("a") is None
        assert state.get("b") == {"x": [1, 2]}
        assert list(state.scan()) == [("b", {"x": [1, 2]}), ("c", "z")]

    def test_state_commit_rejects_stale_cursor(self, state):
        state.commit(0, 1, {"a": 1}, set())

        with pytest.raises(CursorConflict):
            state.commit(0, 2, {"a": 2}, set())
        assert state.get("a") == 1 and state.cursor() == 1

    def test_state_values_are_copied(self, state):
        value = {"n": [1]}
        state.commit(0, 0, {"k": value}, set())
        value["n"].append(2)
        state.get("k")["n"].append(3)

        assert state.get("k") == {"n": [1]}

    def test_state_scan_filters_by_prefix(self, state):
        state.commit(0, 0, {"cand/a": 1, "cand/b": 2, "seen/a": 3}, set())

        assert [k for k, _ in state.scan("cand/")] == ["cand/a", "cand/b"]

    def test_state_rejects_non_json_values(self, state):
        with pytest.raises(TypeError):
            state.commit(0, 0, {"k": object()}, set())

    def test_state_reset_drops_keys_and_cursor_and_records_version(self, state):
        assert state.version() is None
        state.commit(0, 4, {"a": 1}, set())

        state.reset("2")

        assert (state.version(), state.cursor(), list(state.scan())) == ("2", 0, [])


class _Counting:
    """Records each delivery in `seen` and in its state, promotes each event."""

    def __init__(self, seen, state_version=None, fail=False, fail_on=None):
        self.seen = seen
        if state_version is not None:
            self.state_version = state_version
        self.fail = fail
        self.fail_on = fail_on  # an event id that always fails

    def observe(self, event, ctx):
        self.seen.append(event.id)
        ctx.state.put(f"{getattr(self, 'state_version', 'v0')}/{event.id}", True)
        if self.fail or event.id == self.fail_on:
            raise RuntimeError("boom")
        item = Knowledge(id=f"k-{event.id}", kind="knowledge", title=event.id, body=event.text,
                         scope="personal", owner=event.user, sources=(event.id,))
        return [Promote(item)]

    def tick(self, now, ctx):
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return []


class DatabaseContract:
    """make(tmp_path) returns a zero-argument opener. Each call to it is one
    more process (connection) on the same storage."""

    def make(self, tmp_path) -> Any:
        raise NotImplementedError

    @pytest.fixture
    def open_db(self, tmp_path):
        return self.make(tmp_path)

    @staticmethod
    def runtime(open_db, engine, name="engine", max_failures=None):
        db = open_db()
        return Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction, max_failures)

    def test_two_processes_deliver_each_event_once(self, open_db):
        seen = []
        a = self.runtime(open_db, _Counting(seen))
        b = self.runtime(open_db, _Counting(seen))

        a.append(ev("e1", "one"))
        b.append(ev("e2", "two"))
        a.sync()
        b.sync()

        assert seen == ["e1", "e2"]
        assert [k.id for k in a.knowledge.all()] == ["k-e1", "k-e2"]
        assert a.state.cursor() == b.state.cursor() == 2

    def test_failed_delivery_commits_nothing_and_is_retried(self, open_db):
        seen = []
        engine = _Counting(seen, fail=True)
        rt = self.runtime(open_db, engine)

        with pytest.raises(RuntimeError):
            rt.append(ev("e1", "x"))
        assert (rt.state.cursor(), list(rt.state.scan()), rt.knowledge.all()) == (0, [], [])
        assert [e.id for e in rt.events.get(["e1"])] == ["e1"]  # the event itself stays appended

        engine.fail = False
        rt.sync()
        assert seen == ["e1", "e1"]
        assert (rt.state.cursor(), [k.id for k in rt.knowledge.all()]) == (1, ["k-e1"])

    def test_without_max_failures_a_failing_event_blocks_delivery(self, open_db):
        seen = []
        rt = self.runtime(open_db, _Counting(seen, fail_on="e1"))

        with pytest.raises(RuntimeError):
            rt.append(ev("e1", "x"))
        for _ in range(4):
            with pytest.raises(RuntimeError):
                rt.sync()
        with pytest.raises(RuntimeError):
            rt.append(ev("e2", "y"))

        assert seen == ["e1"] * 6
        assert (rt.state.cursor(), rt.knowledge.all(), rt.skipped) == (0, [], [])

    def test_max_failures_skips_the_event_and_delivers_the_rest(self, open_db):
        seen = []
        rt = self.runtime(open_db, _Counting(seen, fail_on="e1"), max_failures=3)

        with pytest.raises(RuntimeError):
            rt.append(ev("e1", "x"))
        with pytest.raises(RuntimeError):
            rt.sync()
        rt.append(ev("e2", "y"))  # third attempt on e1 gives up, then e2 is delivered

        assert seen == ["e1", "e1", "e1", "e2"]
        assert [(s.seq, s.event_id, s.attempts, s.error) for s in rt.skipped] == [(1, "e1", 3, "RuntimeError: boom")]
        assert rt.state.cursor() == 2
        assert [k for k, _ in rt.state.scan()] == ["v0/e2"]  # nothing e1 wrote was kept
        assert [k.id for k in rt.knowledge.all()] == ["k-e2"]

        later = []
        self.runtime(open_db, _Counting(later, fail_on="e1"), max_failures=3).sync()
        assert later == []  # another process does not retry the skipped event

    def test_state_knowledge_and_cursor_survive_reopening(self, open_db):
        self.runtime(open_db, _Counting([])).append(ev("e1", "x"))

        seen = []
        again = self.runtime(open_db, _Counting(seen))
        again.sync()

        assert seen == []
        assert [k for k, _ in again.state.scan()] == ["v0/e1"]
        assert [k.id for k in again.knowledge.all()] == ["k-e1"]

    def test_changed_state_version_drops_state_and_redelivers(self, open_db):
        rt = self.runtime(open_db, _Counting([], state_version="1"))
        rt.append(ev("e1", "x"))
        rt.append(ev("e2", "y"))

        seen = []
        upgraded = self.runtime(open_db, _Counting(seen, state_version="2"))
        upgraded.sync()

        assert seen == ["e1", "e2"]
        assert [k for k, _ in upgraded.state.scan()] == ["2/e1", "2/e2"]
        assert upgraded.state.cursor() == 2

        later = []
        self.runtime(open_db, _Counting(later, state_version="2")).sync()
        assert later == []
