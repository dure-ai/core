from datetime import datetime, timezone

import pytest

from cortex_mem.contract import Caller, Engine, EngineFactory, Event, Handoff, Knowledge
from cortex_mem.engines.baseline import NoMemoryEngine, PromoteAllEngine
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryEventStore, InMemoryKnowledgeBase, InMemoryStateBackend

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)
FACTORIES: list[EngineFactory] = [NoMemoryEngine, PromoteAllEngine]


def ev(id, text="", kind="record", user="u1", team="t1", scope="personal", at=T0, **kw):
    return Event(id=id, kind=kind, at=at, user=user, team=team, session="s1", scope=scope, text=text, **kw)


def caller(user="u1", team="t1"):
    return Caller(user=user, team=team, session="s2", at=T0)


def runtime(factory=PromoteAllEngine):
    return Runtime(factory({}), InMemoryEventStore(), InMemoryKnowledgeBase(), InMemoryStateBackend())


@pytest.mark.parametrize("factory", FACTORIES)
def test_baselines_conform_to_engine(factory):
    assert isinstance(factory({}), Engine)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        ev("e1", at=datetime(2026, 10, 19))
    with pytest.raises(ValueError):
        Caller(user="u1", team="t1", session="s1", at=datetime(2026, 10, 19))


def test_observe_does_not_change_stored_conditions():
    """sunm2n's reproduction on PR #1: an engine must not be able to change a stored event."""
    store = InMemoryEventStore()
    event = ev("e1", conditions={"service": "A"})
    store.append(event)

    with pytest.raises(TypeError):
        event.conditions["service"] = "B"
    assert store.get(["e1"])[0].conditions == {"service": "A"}


def test_conditions_are_copied_from_the_callers_dict():
    conditions = {"service": "A"}
    event = ev("e1", conditions=conditions)
    conditions["service"] = "B"

    assert event.conditions == {"service": "A"}


@pytest.mark.parametrize("cls", [Knowledge, Handoff])
def test_knowledge_and_handoff_conditions_are_read_only(cls):
    if cls is Knowledge:
        obj = Knowledge(id="k", kind="knowledge", title="t", body="b", scope="personal",
                        owner="u1", sources=("e1",), conditions={"a": "1"})
    else:
        obj = Handoff(id="h", content="c", sources=("e1",), conditions={"a": "1"})
    with pytest.raises(TypeError):
        obj.conditions["a"] = "2"


def test_promote_all_promotes_each_record_as_widely_as_it_was_recorded():
    rt = runtime()
    rt.append(ev("e1", "cloud sql needs a proxy", scope="team"))
    rt.append(ev("e2", "my editor uses tabs", scope="personal"))
    rt.append(ev("r1", "cloud sql", kind="recall"))

    assert [(i.id, i.scope, i.owner, i.sources) for i in rt.knowledge.all()] == [
        ("k-e1", "team", "t1", ("e1",)),
        ("k-e2", "personal", "u1", ("e2",)),
    ]
    assert rt.rejections == []


def test_promote_all_hands_off_knowledge_with_resolvable_sources():
    rt = runtime()
    rt.append(ev("e1", "cloud sql needs a proxy"))
    rt.append(ev("e2", "redis is fine"))

    handoffs = rt.handoff("cloud sql proxy", caller(), "personal")

    assert [h.id for h in handoffs] == ["k-e1"]
    for h in handoffs:
        assert [e.id for e in rt.events.get(h.sources)] == list(h.sources)


def test_no_memory_hands_off_nothing():
    rt = runtime(NoMemoryEngine)
    rt.append(ev("e1", "cloud sql needs a proxy"))

    assert rt.handoff("cloud sql", caller(), "personal") == []
    assert rt.knowledge.all() == []


def test_handoff_is_deterministic_for_same_event_sequence():
    def run():
        rt = runtime()
        for i in range(5):
            rt.append(ev(f"e{i}", "deploy uses tags"))
        return [h.id for h in rt.handoff("deploy tags", caller(), "personal", k=3)]

    assert run() == run() == ["k-e0", "k-e1", "k-e2"]


def test_recall_appends_recall_event_with_handed_off_ids():
    rt = runtime()
    rt.append(ev("e1", "deploy uses release tags"))

    handoffs = rt.recall("deploy tags", caller(), "personal", event_id="r1")

    [recall] = rt.events.get(["r1"])
    assert (recall.kind, recall.text, recall.refs) == ("recall", "deploy tags", tuple(h.id for h in handoffs))
    assert (recall.user, recall.team, recall.session, recall.at) == ("u1", "t1", "s2", T0)


def test_recall_event_stays_personal_whatever_scope_was_searched():
    rt = runtime()
    rt.append(ev("e1", "deploy uses release tags", scope="org"))

    rt.recall("deploy tags", caller(), "org", event_id="r1")

    assert rt.events.get(["r1"])[0].scope == "personal"
    assert rt.events.search("deploy tags", caller(user="u2"), "org", kinds={"recall"}) == []
