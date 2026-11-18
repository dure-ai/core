"""Conformance suites for store implementations. A store passes the contract
when a subclass of the matching suite passes:

    class TestSqliteEvents(EventStoreContract):
        def make(self):
            return SqliteDatabase(":memory:").events

DatabaseContract runs the runtime over a whole Database, so delivery
(exactly once, all or nothing, state versions) is checked per backend too.

Needs pytest; import it from tests only.
"""

from array import array
from datetime import datetime, timezone
from typing import Any

import pytest

from .access import owner_for
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

    def test_get_finds_nothing_for_ids_that_are_not_strings(self, store):
        store.append(ev("1", "a"))
        store.append(ev("e2", "b"))

        assert store.get([1, 2, None]) == []  # an engine bug, not a store error
        assert [e.id for e in store.get(["e2", 1, "1"])] == ["e2", "1"]

    def test_get_round_trips_every_field(self, store):
        original = Event(
            id="e1", kind="outcome", at=T0, user="u1", team="t1", session="s1", scope="team",
            text="ci passed", agent="claude-code", signal="test", outcome="success",
            conditions={"service": "a"}, refs=("h1", "h2"), repo="org/repo", rev="abc123", text_redacted=True,
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

    def test_search_ranks_a_rare_word_above_a_common_one(self, store):
        for i in range(3):
            store.append(ev(f"common{i}", "deploy notes"))
        store.append(ev("rare", "canary notes"))

        assert next(e.id for e, _ in store.search("deploy canary", caller(), "personal")) == "rare"

    def test_search_matches_korean_with_a_particle_attached(self, store):
        store.append(ev("e1", "배포가 실패했다"))

        assert [e.id for e, _ in store.search("배포", caller(), "personal")] == ["e1"]

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

    def test_knowledge_get_finds_nothing_for_ids_that_are_not_strings(self, kb):
        kb.put(item("1"))
        kb.put(item("k2"))

        assert kb.get([1, 2, None]) == []  # an engine bug, not a store error
        assert [i.id for i in kb.get(["k2", 1, "1"])] == ["k2", "1"]

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

    def test_knowledge_search_ranks_a_rare_word_above_a_common_one(self, kb):
        for i in range(3):
            kb.put(item(f"common{i}", "deploy notes"))
        kb.put(item("rare", "canary notes"))

        assert next(i.id for i, _ in kb.search("deploy canary", caller(), "team")) == "rare"

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
        state.commit(0, 3, {"a": 1, "b": {"x": [1, 2]}}, set(), epoch=state.epoch())
        state.commit(3, 5, {"c": "z"}, {"a"}, epoch=state.epoch())

        assert state.cursor() == 5
        assert state.get("a") is None
        assert state.get("b") == {"x": [1, 2]}
        assert list(state.scan()) == [("b", {"x": [1, 2]}), ("c", "z")]

    def test_state_commit_rejects_stale_cursor(self, state):
        state.commit(0, 1, {"a": 1}, set(), epoch=state.epoch())

        with pytest.raises(CursorConflict):
            state.commit(0, 2, {"a": 2}, set(), epoch=state.epoch())
        assert state.get("a") == 1 and state.cursor() == 1

    def test_state_values_are_copied(self, state):
        value = {"n": [1]}
        state.commit(0, 0, {"k": value}, set(), epoch=state.epoch())
        value["n"].append(2)
        state.get("k")["n"].append(3)

        assert state.get("k") == {"n": [1]}

    def test_state_scan_filters_by_prefix(self, state):
        state.commit(0, 0, {"cand/a": 1, "cand/b": 2, "seen/a": 3}, set(), epoch=state.epoch())

        assert [k for k, _ in state.scan("cand/")] == ["cand/a", "cand/b"]

    def test_state_rejects_non_json_values(self, state):
        with pytest.raises(TypeError):
            state.commit(0, 0, {"k": object()}, set(), epoch=state.epoch())

    def test_state_reset_drops_keys_and_cursor_and_records_version(self, state):
        assert state.version() is None
        state.commit(0, 4, {"a": 1}, set(), epoch=state.epoch())

        state.reset("2")

        assert (state.version(), state.cursor(), list(state.scan())) == ("2", 0, [])

    def test_epoch_starts_at_zero_and_commit_bumps_it(self, state):
        assert state.epoch() == 0

        state.commit(0, 1, {"a": 1}, set(), epoch=0)
        assert state.epoch() == 1
        state.commit(1, 1, {}, set(), epoch=1)  # a tick: the cursor stays, the epoch moves
        assert state.epoch() == 2

    def test_reset_bumps_the_epoch(self, state):
        state.reset("1")  # no row yet: created at epoch 1
        assert state.epoch() == 1

        state.commit(0, 1, {"a": 1}, set(), epoch=1)
        state.reset("2")
        assert state.epoch() == 3

    def test_two_commits_with_the_same_cursor_and_epoch_only_one_wins(self, state):
        state.commit(0, 2, {"seed": 1}, set(), epoch=state.epoch())
        cursor, epoch = state.cursor(), state.epoch()  # both ticks read this

        state.commit(cursor, cursor, {"decided": "first"}, set(), epoch=epoch)
        with pytest.raises(CursorConflict):
            state.commit(cursor, cursor, {"decided": "second", "extra": 1}, {"seed"}, epoch=epoch)

        assert state.get("decided") == "first"
        assert state.get("extra") is None and state.get("seed") == 1
        assert state.cursor() == cursor and state.epoch() == epoch + 1

    def test_a_commit_after_a_reset_is_refused_even_if_the_cursor_matches(self, state):
        state.reset("1")
        cursor, epoch = state.cursor(), state.epoch()
        assert cursor == 0

        state.reset("2")  # another handle resets the engine: the cursor is 0 again
        assert state.cursor() == cursor

        with pytest.raises(CursorConflict):
            state.commit(0, 1, {"a": 1}, set(), epoch=epoch)

        assert state.get("a") is None and state.cursor() == 0
        assert state.version() == "2" and state.epoch() == epoch + 1

    def test_a_refused_commit_writes_nothing(self, state):
        state.reset("v")
        state.commit(0, 3, {"keep": 1, "gone": 2}, set(), epoch=state.epoch())
        before = (list(state.scan()), state.cursor(), state.version(), state.epoch())

        with pytest.raises(CursorConflict):  # wrong cursor
            state.commit(0, 9, {"new": 1, "keep": 5}, {"gone"}, epoch=state.epoch())
        with pytest.raises(CursorConflict):  # wrong epoch
            state.commit(3, 9, {"new": 1, "keep": 5}, {"gone"}, epoch=state.epoch() - 1)

        assert (list(state.scan()), state.cursor(), state.version(), state.epoch()) == before


def f32(values):
    """The values as a store keeps them: rounded to float32."""
    return array("f", values).tolist()


SCORE_TOLERANCE = 1e-6  # stores may differ in the last float32 digits (PG computes in SQL)


class VectorBackendContract(_Contract):
    """make() returns a Database-like object with .events, .knowledge and
    .vectors on one storage: the similarity methods judge visibility from the
    events and knowledge they sit next to."""

    @pytest.fixture
    def db(self):
        return self.make()

    @pytest.fixture
    def vb(self, db):
        return db.vectors

    @staticmethod
    def index_event(db, event, vector, seq):
        """Append event, store its vector and group it on its own at position seq."""
        db.events.append(event)
        db.vectors.put_vector("event", event.id, vector)
        db.vectors.put_group(event.id, event.id, event.scope, owner_for(event), seq)

    @staticmethod
    def scores_match(got, want):
        assert [id for id, _ in got] == [id for id, _ in want]
        for (_, a), (_, b) in zip(got, want, strict=True):
            assert a == pytest.approx(b, abs=SCORE_TOLERANCE)

    def test_kind_names_the_store(self, vb):
        assert vb.kind in ("memory", "sqlite", "postgres")

    def test_nearest_member_is_the_closest_in_the_cell_and_ties_go_to_the_earlier(self, vb):
        # Inserted out of seq order: ties follow the seq given to put_group.
        rows = [("late", [1.0, 0.0], 4), ("early", [1.0, 0.0], 2), ("up", [0.0, 1.0], 1),
                ("diag", [0.6, 0.8], 3), ("away", [-1.0, 0.0], 5)]  # fmt: skip
        for id, vector, seq in rows:
            vb.put_vector("event", id, vector)
            vb.put_group(id, id, "team", "t1", seq)
        vb.put_group("novector", "novector", "team", "t1", 6)  # grouped but no vector: skipped

        got = vb.nearest_member("team", "t1", [1.0, 0.0], 10)
        assert got[0] == "early" and got[1] == pytest.approx(1.0, abs=SCORE_TOLERANCE)
        got = vb.nearest_member("team", "t1", [0.0, 1.0], 10)
        assert got[0] == "up" and got[1] == pytest.approx(1.0, abs=SCORE_TOLERANCE)
        got = vb.nearest_member("team", "t1", [0.6, 0.8], 10)
        assert got[0] == "diag" and got[1] == pytest.approx(1.0, abs=SCORE_TOLERANCE)
        got = vb.nearest_member("team", "t1", [-1.0, 0.0], 3)  # "away" is past upto
        assert got[0] == "up" and got[1] == pytest.approx(0.0, abs=SCORE_TOLERANCE)
        # The closest even when it is not close: the caller applies its threshold.
        vb.put_vector("event", "only", [-1.0, 0.0])
        vb.put_group("only", "only", "personal", "u1", 7)
        got = vb.nearest_member("personal", "u1", [1.0, 0.0], 10)
        assert got[0] == "only" and got[1] == pytest.approx(-1.0, abs=SCORE_TOLERANCE)
        assert vb.nearest_member("personal", "u2", [1.0, 0.0], 10) is None

    def test_nearest_member_ignores_other_cells_and_members_after_upto(self, vb):
        rows = [("a", [0.0, 1.0], "team", "t1", 1), ("c", [1.0, 0.0], "team", "t2", 2),
                ("d", [1.0, 0.0], "personal", "t1", 3), ("e", [1.0, 0.0], "org", "*", 4),
                ("b", [1.0, 0.0], "team", "t1", 5)]  # fmt: skip
        for id, vector, scope, owner, seq in rows:
            vb.put_vector("event", id, vector)
            vb.put_group(id, id, scope, owner, seq)

        assert vb.nearest_member("team", "t1", [1.0, 0.0], 0) is None
        assert vb.nearest_member("team", "t1", [1.0, 0.0], 4)[0] == "a"
        assert vb.nearest_member("team", "t1", [1.0, 0.0], 5)[0] == "b"
        assert vb.nearest_member("org", "*", [1.0, 0.0], 3) is None
        assert vb.nearest_member("org", "*", [1.0, 0.0], 4)[0] == "e"

    def test_search_events_visibility_matrix(self, db, vb):
        rows = [
            ev("p-u1", user="u1", team="t1", scope="personal"),
            ev("p-u2", user="u2", team="t1", scope="personal"),
            ev("t-t1", user="u2", team="t1", scope="team"),
            ev("t-t2", user="u3", team="t2", scope="team"),
            ev("o-u9", user="u9", team="t9", scope="org"),
            ev("o-u3", user="u3", team="t2", scope="org"),
        ]
        for seq, e in enumerate(rows, start=1):
            self.index_event(db, e, [1.0, 0.0], seq)
        expected = {
            (("u1", "t1"), "personal"): ["p-u1"],
            (("u1", "t1"), "team"): ["p-u1", "t-t1"],
            (("u1", "t1"), "org"): ["p-u1", "t-t1", "o-u9", "o-u3"],
            (("u2", "t1"), "personal"): ["p-u2", "t-t1"],
            (("u2", "t1"), "team"): ["p-u2", "t-t1"],
            (("u2", "t1"), "org"): ["p-u2", "t-t1", "o-u9", "o-u3"],
            (("u3", "t2"), "personal"): ["t-t2", "o-u3"],
            (("u3", "t2"), "team"): ["t-t2", "o-u3"],
            (("u3", "t2"), "org"): ["t-t2", "o-u9", "o-u3"],
        }
        got = {
            (who, scope): [id for id, _ in vb.search_events([1.0, 0.0], caller(*who), scope, 50)]
            for who, scope in expected
        }
        assert got == expected

    def test_search_events_respects_kinds_upto_exclude_k_and_drops_nonpositive(self, db, vb):
        rows = [("a", "record", [1.0, 0.0]), ("b", "recall", [0.8, 0.6]), ("c", "record", [0.0, 1.0]),
                ("d", "record", [-1.0, 0.0]), ("e", "record", [0.8, 0.6]), ("f", "record", [1.0, 0.0])]  # fmt: skip
        for seq, (id, kind, vector) in enumerate(rows, start=1):
            self.index_event(db, ev(id, kind=kind), vector, seq)
        db.events.append(ev("loose"))
        vb.put_vector("event", "loose", [1.0, 0.0])  # no group: not a candidate
        vb.put_vector("event", "ghost", [1.0, 0.0])
        vb.put_group("ghost", "ghost", "personal", "u1", 8)  # no event: not a candidate
        q, me = [1.0, 0.0], caller()

        def search(**kw):
            return [id for id, _ in vb.search_events(q, me, "personal", kw.pop("k", 20), **kw)]

        self.scores_match(
            vb.search_events(q, me, "personal", 20), [("a", 1.0), ("f", 1.0), ("b", 0.8), ("e", 0.8)]
        )
        assert search(kinds=None, upto=None) == ["a", "f", "b", "e"]
        assert search(kinds={"record"}) == ["a", "f", "e"]
        assert search(kinds=["recall"]) == ["b"]
        assert search(kinds=()) == []
        assert search(upto=5) == ["a", "b", "e"]
        assert search(upto=0) == []
        assert search(exclude="a") == ["f", "b", "e"]
        assert search(exclude="a", upto=5, kinds={"record"}) == ["e"]
        assert search(k=2) == ["a", "f"]
        assert search(k=0) == []
        assert search(k=-1) == []

    def test_search_knowledge_visibility_and_id_ties(self, db, vb):
        rows = [("k1", "personal", "u1", [1.0, 0.0]), ("k2", "personal", "u2", [1.0, 0.0]),
                ("k3", "team", "t1", [1.0, 0.0]), ("k4", "team", "t2", [1.0, 0.0]),
                ("k5", "org", "*", [1.0, 0.0]), ("k6", "org", "*", [0.6, 0.8]),
                ("k7", "org", "*", [0.0, 1.0])]  # fmt: skip
        for id, scope, owner, vector in reversed(rows):  # stored out of id order
            db.knowledge.put(item(id, scope=scope, owner=owner))
            vb.put_vector("knowledge", id, vector)
        vb.put_vector("knowledge", "ghost", [1.0, 0.0])  # not in the knowledge base
        expected = {
            (("u1", "t1"), "personal"): ["k1"],
            (("u1", "t1"), "team"): ["k1", "k3"],
            (("u1", "t1"), "org"): ["k1", "k3", "k5", "k6"],
            (("u2", "t1"), "personal"): ["k2"],
            (("u2", "t1"), "team"): ["k2", "k3"],
            (("u2", "t1"), "org"): ["k2", "k3", "k5", "k6"],
            (("u3", "t2"), "personal"): [],
            (("u3", "t2"), "team"): ["k4"],
            (("u3", "t2"), "org"): ["k4", "k5", "k6"],
        }
        got = {
            (who, scope): [id for id, _ in vb.search_knowledge([1.0, 0.0], caller(*who), scope, 50)]
            for who, scope in expected
        }
        assert got == expected
        self.scores_match(
            vb.search_knowledge([1.0, 0.0], caller(), "org", 3), [("k1", 1.0), ("k3", 1.0), ("k5", 1.0)]
        )
        self.scores_match(vb.search_knowledge([0.6, 0.8], caller("u3", "t2"), "org", 50),
                          [("k6", 1.0), ("k7", 0.8), ("k4", 0.6), ("k5", 0.6)])  # fmt: skip
        assert vb.search_knowledge([1.0, 0.0], caller(), "org", 0) == []

    def test_meta_starts_empty_and_reset_records_it(self, vb):
        assert vb.meta() is None
        assert vb.indexed() == 0

        meta = {"model": "m1", "group_threshold": 0.8}
        vb.reset(meta)

        assert vb.meta() == meta
        vb.meta()["model"] = "changed"
        assert vb.meta() == meta

    def test_vectors_round_trip_at_float32_precision(self, vb):
        values = [0.1, 1 / 3, -2.5]
        want = f32(values)
        assert want != values  # the rounding is visible
        vb.put_vector("event", "e1", values)
        vb.put_vector("knowledge", "k1", values)

        got = vb.vector("event", "e1")
        assert got == want
        assert vb.vector("knowledge", "k1") == want
        assert list(vb.vectors("event")) == [("e1", want)]
        assert all(type(x) is float for x in got)
        assert vb.vector("event", "missing") is None
        assert vb.vector("knowledge", "e1") is None
        got.append(9.0)
        assert vb.vector("event", "e1") == want

    def test_event_vectors_keep_insertion_order_and_knowledge_vectors_sort_by_id(
        self, vb
    ):
        for id in ("e3", "e1", "e2"):
            vb.put_vector("event", id, [1.0])
        for id in ("k3", "k1", "k2"):
            vb.put_vector("knowledge", id, [2.0])

        assert [i for i, _ in vb.vectors("event")] == ["e3", "e1", "e2"]
        assert [i for i, _ in vb.vectors("knowledge")] == ["k1", "k2", "k3"]
        for _, vec in vb.vectors("event"):
            vec.append(5.0)
        assert vb.vector("event", "e3") == [1.0]

    def test_put_vector_replaces_the_same_id(self, vb):
        # One dimension throughout: PostgreSQL refuses mixed dimensions in an index version.
        vb.put_vector("event", "e1", [1.0, 0.0])
        vb.put_vector("event", "e2", [2.0, 0.0])
        vb.put_vector("event", "e1", [9.0, 8.0])

        assert vb.vector("event", "e1") == [9.0, 8.0]
        assert list(vb.vectors("event")) == [("e1", [9.0, 8.0]), ("e2", [2.0, 0.0])]

    def test_delete_vectors_removes_only_the_named_ids(self, vb):
        for id in ("e1", "e2", "e3"):
            vb.put_vector("event", id, [1.0])
        vb.put_vector("knowledge", "e1", [3.0])
        vb.put_group("e1", "g1", "team", "t1", 1)

        vb.delete_vectors("event", {"e1", "nope"})

        assert [i for i, _ in vb.vectors("event")] == ["e2", "e3"]
        assert vb.vector("knowledge", "e1") == [3.0]
        assert vb.group_of("e1") == "g1"

    def test_groups_members_and_partition_are_in_seq_order(self, vb):
        vb.put_group("e3", "g1", "team", "t1", 3)
        vb.put_group("e1", "g1", "team", "t1", 1)
        vb.put_group("e2", "g1", "team", "t1", 2)

        assert vb.members("g1") == ["e1", "e2", "e3"]
        assert vb.partition("team", "t1") == ["e1", "e2", "e3"]
        assert vb.group_of("e2") == "g1"
        assert vb.group_of("unknown") is None
        assert vb.members("unknown") == []

    def test_partition_is_per_scope_and_owner(self, vb):
        vb.put_group("a", "g1", "team", "t1", 1)
        vb.put_group("b", "g2", "team", "t2", 2)
        vb.put_group("c", "g3", "personal", "u1", 3)

        assert vb.partition("team", "t1") == ["a"]
        assert vb.partition("team", "t2") == ["b"]
        assert vb.partition("personal", "u1") == ["c"]
        assert vb.partition("org", "t1") == []

    def test_put_group_again_moves_the_event_to_its_new_cell_and_seq(self, vb):
        for id, seq in (("a", 1), ("b", 2)):
            vb.put_vector("event", id, [1.0, 0.0])
            vb.put_group(id, id, "team", "t1", seq)

        vb.put_group("a", "a", "team", "t1", 3)  # same cell, later seq
        assert vb.partition("team", "t1") == ["b", "a"]
        assert vb.nearest_member("team", "t1", [1.0, 0.0], 3)[0] == "b"
        vb.put_group("a", "a", "team", "t2", 3)  # another cell
        assert vb.partition("team", "t1") == ["b"]
        assert vb.partition("team", "t2") == ["a"]
        assert vb.nearest_member("team", "t2", [1.0, 0.0], 3)[0] == "a"

    def test_reset_clears_vectors_groups_and_cursor_but_keeps_the_cache(self, vb):
        vb.reset({"model": "m1", "group_threshold": 0.8})
        vb.put_vector("event", "e1", [1.0])
        vb.put_vector("knowledge", "k1", [1.0])
        vb.put_group("e1", "g1", "team", "t1", 1)
        vb.cache("m1", "d1", [0.5])
        vb.set_indexed(0, 7, vb.generation())

        vb.reset({"model": "m2", "group_threshold": 0.9})

        assert list(vb.vectors("event")) == []
        assert list(vb.vectors("knowledge")) == []
        assert vb.group_of("e1") is None
        assert vb.members("g1") == []
        assert vb.partition("team", "t1") == []
        assert vb.indexed() == 0
        assert vb.meta() == {"model": "m2", "group_threshold": 0.9}
        assert vb.cached("m1", "d1") == [0.5]

    def test_cache_is_per_model(self, vb):
        assert vb.cached("m1", "d1") is None
        vb.cache("m1", "d1", [0.1, 0.2])

        assert vb.cached("m1", "d1") == f32([0.1, 0.2])  # the cache is float32 too
        assert vb.cached("m2", "d1") is None
        assert vb.cached("m1", "d2") is None
        vb.cached("m1", "d1").append(1.0)
        assert vb.cached("m1", "d1") == f32([0.1, 0.2])

    def test_members_and_event_vectors_up_to_a_position(self, vb):
        for seq, id in enumerate(["e1", "e2", "e3"], start=1):
            vb.put_vector("event", id, [float(seq)])
            vb.put_group(id, "g1", "team", "t1", seq * 10)  # positions 10, 20, 30
        vb.put_vector("knowledge", "k1", [1.0])

        assert vb.members("g1", upto=20) == ["e1", "e2"]
        assert vb.members("g1", upto=9) == []
        assert vb.members("g1", upto=None) == vb.members("g1") == ["e1", "e2", "e3"]
        assert [i for i, _ in vb.vectors("event", upto=20)] == ["e1", "e2"]
        assert list(vb.vectors("event", upto=30)) == list(vb.vectors("event"))
        assert list(vb.vectors("event", upto=0)) == []
        with pytest.raises(ValueError):
            list(vb.vectors("knowledge", upto=20))

    def test_set_indexed_moves_the_cursor(self, vb):
        g = vb.generation()
        vb.set_indexed(0, 5, g)
        assert vb.indexed() == 5
        vb.set_indexed(5, 9, g)
        assert vb.indexed() == 9

    def test_generation_starts_at_zero_and_reset_bumps_it(self, vb):
        assert vb.generation() == 0
        meta = {"model": "m1", "group_threshold": 0.8}
        vb.reset(meta)
        assert vb.generation() == 1
        assert vb.meta() == meta  # the generation is not part of meta
        vb.reset({"model": "m2", "group_threshold": 0.9})
        assert vb.generation() == 2

    def test_set_indexed_checks_position_and_generation(self, vb):
        vb.reset({"model": "m1", "group_threshold": 0.8})
        g = vb.generation()
        vb.set_indexed(0, 3, g)
        assert vb.indexed() == 3

        with pytest.raises(CursorConflict):
            vb.set_indexed(2, 4, g)  # the position is 3, not 2
        with pytest.raises(CursorConflict):
            vb.set_indexed(3, 4, g + 1)  # another generation
        with pytest.raises(CursorConflict):
            vb.set_indexed(3, 4, g - 1)
        assert vb.indexed() == 3
        assert vb.generation() == g

        vb.set_indexed(3, 4, g)
        assert vb.indexed() == 4
        vb.reset({"model": "m1", "group_threshold": 0.8})  # a reset moves the generation on
        with pytest.raises(CursorConflict):
            vb.set_indexed(0, 1, g)
        assert vb.indexed() == 0
        vb.set_indexed(0, 1, vb.generation())
        assert vb.indexed() == 1

    def test_knowledge_vectors_keep_their_digest(self, vb):
        assert vb.knowledge_digests() == {}
        vb.put_vector("knowledge", "k1", [1.0, 0.0], digest="d1")
        vb.put_vector("knowledge", "k2", [0.0, 1.0], digest="d2")
        vb.put_vector("knowledge", "k3", [1.0, 1.0])  # no digest: not listed
        assert vb.knowledge_digests() == {"k1": "d1", "k2": "d2"}
        assert vb.vector("knowledge", "k3") == f32([1.0, 1.0])

        vb.put_vector("knowledge", "k1", [0.5, 0.5], digest="d1b")  # replaced with its digest
        assert vb.knowledge_digests() == {"k1": "d1b", "k2": "d2"}
        assert vb.vector("knowledge", "k1") == f32([0.5, 0.5])
        vb.put_vector("knowledge", "k2", [0.5, 0.5])  # replaced without one: the old digest goes
        assert vb.knowledge_digests() == {"k1": "d1b"}

        vb.put_vector("event", "e1", [1.0, 0.0], digest="ignored")
        assert vb.knowledge_digests() == {"k1": "d1b"}
        assert vb.vector("event", "e1") == f32([1.0, 0.0])

        vb.put_vector("knowledge", "k2", [0.0, 1.0], digest="d2")
        vb.delete_vectors("knowledge", ["k1", "nope"])
        assert vb.knowledge_digests() == {"k2": "d2"}
        vb.put_vector("knowledge", "k1", [1.0, 0.0])  # a new vector under a deleted id has no old digest
        assert vb.knowledge_digests() == {"k2": "d2"}

        vb.cache("m1", "d1", [0.5])
        vb.reset({"model": "m1", "group_threshold": 0.8})
        assert vb.knowledge_digests() == {}
        assert vb.cached("m1", "d1") == [0.5]


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
    more process (connection) on the same storage.

    `rolls_back`: whether transaction() undoes the writes made inside it when
    it raises. A durable store must; a single-process store whose
    transaction() is nullcontext sets it False (delivery then stays all or
    nothing because the runtime writes only after everything that can fail)."""

    rolls_back = True

    def make(self, tmp_path) -> Any:
        raise NotImplementedError

    @pytest.fixture
    def open_db(self, tmp_path):
        return self.make(tmp_path)

    @staticmethod
    def runtime(open_db, engine, name="engine", max_failures=None):
        db = open_db()
        return Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction, max_failures)

    def test_epoch_survives_reopening(self, open_db):
        first = open_db().state("engine")
        first.reset("v")
        first.commit(0, 2, {"a": 1}, set(), epoch=first.epoch())

        again = open_db().state("engine")

        assert (again.epoch(), again.cursor(), again.get("a")) == (2, 2, 1)
        with pytest.raises(CursorConflict):
            again.commit(2, 3, {}, set(), epoch=1)
        again.commit(2, 3, {}, set(), epoch=2)
        assert open_db().state("engine").epoch() == 3

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

    def test_vector_writes_roll_back_with_the_transaction(self, open_db):
        if not self.rolls_back:
            pytest.skip("this store's transaction() does not roll back")
        db = open_db()
        vb = db.vectors  # every Database has one, embeddings in use or not
        vb.reset({"model": "m1", "group_threshold": 0.9})
        vb.put_vector("event", "e1", [1.0, 0.0])
        vb.put_group("e1", "e1", "team", "t1", 1)
        vb.put_vector("knowledge", "k1", [0.0, 1.0], digest="dk")
        vb.cache("m1", "d1", [0.5, 0.5])
        vb.set_indexed(0, 1, vb.generation())

        def snapshot(v):
            return (
                v.meta(), v.indexed(), list(v.vectors("event")), list(v.vectors("knowledge")),
                v.members("e1"), v.group_of("e2"), v.cached("m1", "d1"), v.cached("m1", "d2"),
                v.generation(), v.knowledge_digests(),
            )

        before = snapshot(vb)
        with pytest.raises(RuntimeError, match="boom"), db.transaction():
            vb.put_vector("event", "e2", [0.0, 1.0])
            vb.put_group("e2", "e1", "team", "t1", 2)
            vb.put_vector("knowledge", "k1", [9.0, 9.0], digest="other")
            vb.delete_vectors("event", ["e1"])
            vb.cache("m1", "d2", [0.1, 0.2])
            vb.set_indexed(1, 2, vb.generation())
            vb.reset({"model": "m2", "group_threshold": 0.5})
            raise RuntimeError("boom")

        assert snapshot(vb) == before
        assert snapshot(open_db().vectors) == before  # another connection sees the same
