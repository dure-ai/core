from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.contract import Caller, Demote, Event, Handoff, Knowledge, Promote
from cortex_mem.engines.baseline import PromoteAllEngine
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryEventStore, InMemoryKnowledgeBase, InMemoryStateBackend

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)


def ev(id, text="", kind="record", user="u1", team="t1", scope="personal", at=T0, **kw):
    return Event(id=id, kind=kind, at=at, user=user, team=team, session="s1", scope=scope, text=text, **kw)


def caller(user="u1", team="t1"):
    return Caller(user=user, team=team, session="s2", at=T0)


def item(id, sources, scope="team", owner="t1", body="deploy uses tags"):
    return Knowledge(id=id, kind="knowledge", title=id, body=body, scope=scope, owner=owner, sources=sources)


class Scripted:
    """Engine whose decisions and handoffs a test sets per event id."""

    def __init__(self, decisions=None, handoffs=(), on_tick=()):
        self.decisions = decisions or {}
        self.handoffs = list(handoffs)
        self.on_tick = list(on_tick)
        self.seen = []

    def observe(self, event, ctx):
        self.seen.append(event.id)
        return self.decisions.get(event.id, [])

    def tick(self, now, ctx):
        return self.on_tick

    def handoff(self, query, caller, scope, ctx, k=6):
        return self.handoffs


def runtime(engine, events=None, kb=None, state=None):
    return Runtime(
        engine,
        events or InMemoryEventStore(),
        kb or InMemoryKnowledgeBase(),
        state or InMemoryStateBackend(),
    )


# ── delivery ────────────────────────────────────────────────────────────────


def test_each_event_is_delivered_once_in_append_order():
    engine = Scripted()
    rt = runtime(engine)
    for id in ["b", "a", "c"]:
        rt.append(ev(id))
    rt.sync()
    rt.handoff("x", caller(), "personal")

    assert engine.seen == ["b", "a", "c"]
    assert rt.state.cursor() == 3


def test_observe_can_read_its_own_event():
    seen = []

    class Probe(Scripted):
        def observe(self, event, ctx):
            seen.extend(ctx.events.get([event.id]))
            return []

    runtime(Probe()).append(ev("e1"))
    assert [e.id for e in seen] == ["e1"]


def test_failed_observe_commits_nothing_and_is_retried():
    class Flaky(Scripted):
        fail = True

        def observe(self, event, ctx):
            ctx.state.put("count", (ctx.state.get("count") or 0) + 1)
            if self.fail:
                raise RuntimeError("boom")
            return [Promote(item("k1", (event.id,), scope="personal", owner="u1"))]

    engine = Flaky()
    rt = runtime(engine)
    with pytest.raises(RuntimeError):
        rt.append(ev("e1"))
    assert (rt.state.cursor(), rt.state.get("count"), rt.knowledge.all()) == (0, None, [])

    engine.fail = False
    rt.sync()
    assert (rt.state.cursor(), rt.state.get("count"), [i.id for i in rt.knowledge.all()]) == (1, 1, ["k1"])


def test_state_written_in_observe_is_visible_next_time():
    class Counter(Scripted):
        def observe(self, event, ctx):
            ctx.state.put("n", (ctx.state.get("n") or 0) + 1)
            return []

    rt = runtime(Counter())
    for i in range(3):
        rt.append(ev(f"e{i}"))
    assert rt.state.get("n") == 3


def test_second_session_sees_first_sessions_record():
    """Two server processes share the stores; B started before A recorded."""
    events, kb, state = InMemoryEventStore(), InMemoryKnowledgeBase(), InMemoryStateBackend()
    session_a = runtime(PromoteAllEngine(), events, kb, state)
    session_b = runtime(PromoteAllEngine(), events, kb, state)

    session_a.append(ev("e1", "cloud sql needs the auth proxy"))

    assert [h.id for h in session_b.handoff("cloud sql proxy", caller(), "personal")] == ["k-e1"]


def test_state_values_are_copied_in_and_out_during_observe():
    """sunm2n's reproduction on PR #1: pending writes must not share objects with the engine."""

    class Mutating(Scripted):
        def observe(self, event, ctx):
            value = {"n": 1}
            ctx.state.put("x", value)
            value["n"] = 2
            ctx.state.get("x")["n"] = 3
            for _, v in ctx.state.scan("x"):
                v["n"] = 4
            return []

    rt = runtime(Mutating())
    rt.append(ev("e1"))

    assert rt.state.get("x") == {"n": 1}


def test_state_values_are_copied_in_during_tick():
    class Mutating(Scripted):
        def tick(self, now, ctx):
            value = {"n": 1}
            ctx.state.put("x", value)
            value["n"] = 2
            return []

    rt = runtime(Mutating())
    rt.tick(T0)

    assert rt.state.get("x") == {"n": 1}


@pytest.mark.parametrize("kind", ["outcome", "human_confirm"])
def test_judgement_without_refs_is_refused(kind):
    engine = Scripted()
    rt = runtime(engine)

    with pytest.raises(ValueError, match="needs refs"):
        rt.append(ev("o1", kind=kind, outcome="failure"))
    assert rt.events.get(["o1"]) == [] and engine.seen == []

    rt.append(ev("o2", kind=kind, outcome="failure", refs=("k1",)))
    assert engine.seen == ["o2"]


def test_oversized_state_value_is_refused():
    class Hoarder(Scripted):
        def observe(self, event, ctx):
            ctx.state.put("everything", "x" * (64 * 1024))
            return []

    rt = runtime(Hoarder())
    with pytest.raises(ValueError, match="split it across keys"):
        rt.append(ev("e1"))
    assert rt.state.cursor() == 0


def test_handoff_cannot_write_state():
    class Writer(Scripted):
        def handoff(self, query, caller, scope, ctx, k=6):
            ctx.state.put("x", 1)
            return []

    with pytest.raises(PermissionError):
        runtime(Writer()).handoff("q", caller(), "personal")


# ── decision checks ─────────────────────────────────────────────────────────


def check(decision, *events):
    """Append events, have the engine return `decision` on the last one."""
    engine = Scripted({events[-1].id: [decision]})
    rt = runtime(engine)
    for e in events:
        rt.append(e)
    return rt


def test_promote_with_visible_sources_is_applied():
    rt = check(Promote(item("k1", ("e1", "e2"))), ev("e1", scope="team"), ev("e2", user="u2", scope="team"))

    assert [i.id for i in rt.knowledge.all()] == ["k1"]
    assert rt.rejections == []


@pytest.mark.parametrize(
    "decision, reason",
    [
        (Promote(item("k1", ())), "no sources"),
        (Promote(item("k1", ("e1", "nope"))), "unknown sources"),
        (Promote(item("", ("e1",))), "empty id"),
        (Demote("missing", "stale", ("e1",)), "no knowledge item"),
        ("not a decision", "not a decision"),
    ],
)
def test_malformed_decisions_are_rejected(decision, reason):
    rt = check(decision, ev("e1", scope="team"))

    assert rt.knowledge.all() == []
    assert len(rt.rejections) == 1 and reason in rt.rejections[0].reason


def test_personal_record_cannot_become_team_knowledge():
    rt = check(Promote(item("k1", ("e1", "e2"))), ev("e1", scope="team"), ev("e2", user="u2", scope="personal"))

    assert rt.knowledge.all() == []
    assert "not visible" in rt.rejections[0].reason


def test_personal_records_of_several_people_cannot_become_team_knowledge():
    """v0.1: cross-checks happen on team records. The handoff filter would
    hide such an item from teammates anyway, since they cannot see its sources."""
    rt = check(Promote(item("k1", ("e1", "e2"))), ev("e1", scope="personal"), ev("e2", user="u2", scope="personal"))

    assert rt.knowledge.all() == []


def test_other_teams_record_cannot_support_this_teams_knowledge():
    rt = check(Promote(item("k1", ("e1",))), ev("e1", user="u3", team="t2", scope="team"))

    assert rt.knowledge.all() == []


def test_org_knowledge_needs_org_sources():
    rejected = check(Promote(item("k1", ("e1",), scope="org", owner="*")), ev("e1", scope="team"))
    accepted = check(Promote(item("k1", ("e1",), scope="org", owner="*")), ev("e1", scope="org"))

    assert rejected.knowledge.all() == []
    assert [i.id for i in accepted.knowledge.all()] == ["k1"]


def test_personal_knowledge_may_cite_its_owners_events():
    rt = check(Promote(item("k1", ("e1", "e2"), scope="personal", owner="u1")),
               ev("e1", scope="personal"), ev("e2", scope="team"))

    assert [i.id for i in rt.knowledge.all()] == ["k1"]


def test_demote_removes_item_and_checks_its_sources():
    engine = Scripted({
        "e1": [Promote(item("k1", ("e1",)))],
        "fail_personal": [Demote("k1", "failed", ("fail_personal",))],
        "fail_team": [Demote("k1", "failed", ("fail_team",))],
    })
    rt = runtime(engine)
    rt.append(ev("e1", scope="team"))
    rt.append(ev("fail_personal", kind="outcome", user="u2", scope="personal", refs=("k1",)))
    assert [i.id for i in rt.knowledge.all()] == ["k1"]

    rt.append(ev("fail_team", kind="outcome", user="u2", scope="team", refs=("k1",)))
    assert rt.knowledge.all() == []
    assert len(rt.rejections) == 1


def test_every_decision_is_logged_with_its_verdict():
    accepted = Promote(item("k1", ("e1",)))
    refused = Promote(item("k2", ("nope",)))
    rt = check(accepted, ev("e1", scope="team"))
    rt.engine.decisions["e2"] = [refused]
    rt.append(ev("e2", scope="team"))

    assert [(v.decision, v.accepted, v.at) for v in rt.verdicts] == [(accepted, True, T0), (refused, False, T0)]
    assert rt.verdicts[0].reason is None and "unknown sources" in rt.verdicts[1].reason
    assert rt.rejections == [rt.verdicts[1]]


def test_decisions_in_one_batch_see_each_other():
    engine = Scripted({"e1": [Promote(item("k1", ("e1",))), Demote("k1", "changed my mind", ("e1",))]})
    rt = runtime(engine)
    rt.append(ev("e1", scope="team"))

    assert rt.knowledge.all() == []
    assert rt.rejections == []


def test_tick_decisions_are_checked_and_applied():
    engine = Scripted({"e1": [Promote(item("k1", ("e1",)))]})
    rt = runtime(engine)
    rt.append(ev("e1", scope="team"))
    engine.on_tick = [Demote("k1", "expired", ("e1",))]

    rt.tick(T0 + timedelta(days=30))

    assert rt.knowledge.all() == []
    assert rt.state.cursor() == 1


# ── handoff filter ──────────────────────────────────────────────────────────


def test_handoff_drops_items_the_caller_cannot_see():
    leaky = Scripted(handoffs=[
        Handoff(id="ok", content="mine", sources=("mine",)),
        Handoff(id="leak", content="their personal note", sources=("theirs",)),
        Handoff(id="no-sources", content="?", sources=()),
        Handoff(id="ghost", content="?", sources=("nope",)),
    ])
    rt = runtime(leaky)
    rt.append(ev("mine", scope="personal"))
    rt.append(ev("theirs", user="u2", scope="personal"))

    assert [h.id for h in rt.handoff("q", caller(), "team")] == ["ok"]


def test_handoff_respects_requested_scope():
    engine = Scripted(handoffs=[Handoff(id="h", content="team note", sources=("e1",))])
    rt = runtime(engine)
    rt.append(ev("e1", user="u2", scope="team"))

    assert rt.handoff("q", caller(), "personal") == []
    assert [h.id for h in rt.handoff("q", caller(), "team")] == ["h"]


def test_handoff_is_cut_to_k():
    engine = Scripted(handoffs=[Handoff(id=f"h{i}", content="x", sources=("e1",)) for i in range(5)])
    rt = runtime(engine)
    rt.append(ev("e1"))

    assert len(rt.handoff("q", caller(), "personal", k=2)) == 2


# ── transactions and concurrent delivery ────────────────────────────────────


def test_each_delivery_runs_in_its_own_transaction():
    log = []

    class Tx:
        def __enter__(self):
            log.append("begin")

        def __exit__(self, *exc):
            log.append("rollback" if exc[0] else "commit")

    engine = Scripted()
    rt = Runtime(engine, InMemoryEventStore(), InMemoryKnowledgeBase(), InMemoryStateBackend(), transaction=Tx)
    rt.events.append(ev("e1"))
    rt.events.append(ev("e2"))
    rt.sync()

    # two deliveries, then one more transaction that finds nothing to deliver
    assert log == ["begin", "commit", "begin", "commit", "begin", "commit"]


def test_event_delivered_by_another_process_is_not_applied_twice():
    events, kb, state = InMemoryEventStore(), InMemoryKnowledgeBase(), InMemoryStateBackend()
    other = runtime(PromoteAllEngine(), events, kb, state)

    class Racing(PromoteAllEngine):
        """While this process is observing, the other one delivers the same event."""

        def observe(self, event, ctx):
            other.sync()
            return [Promote(item("dup", (event.id,), scope="personal", owner="u1"))]

    rt = runtime(Racing(), events, kb, state)
    events.append(ev("e1", "deploy uses tags"))
    rt.sync()

    assert [i.id for i in kb.all()] == ["k-e1"]  # only the winner's decision
    assert state.cursor() == 1 and rt.rejections == []
