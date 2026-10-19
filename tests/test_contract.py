from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.contract import (
    Engine,
    EngineFactory,
    Event,
    Handoff,
    KnowledgeSnapshot,
    QueryContext,
)
from cortex_mem.engines.baseline import NoMemoryEngine, PromoteAllEngine
from cortex_mem.store.memory import InMemoryEpisodeStore

T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)
FACTORIES: list[EngineFactory] = [NoMemoryEngine, PromoteAllEngine]


def ev(id, text="", kind="record", user="u1", scope="personal", at=T0, **kw):
    return Event(id=id, kind=kind, at=at, user=user, session="s1", scope=scope, text=text, **kw)


def ctx(user="u1", scope="personal"):
    return QueryContext(user=user, session="s2", scope=scope)


def feed(store, engine, *events):
    for e in events:
        store.append(e)
        engine.observe(e)


@pytest.mark.parametrize("factory", FACTORIES)
def test_baselines_conform_to_engine_and_snapshot(factory):
    engine = factory(InMemoryEpisodeStore())
    assert isinstance(engine, Engine)
    assert isinstance(engine, KnowledgeSnapshot)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        ev("e1", at=datetime(2026, 10, 19))


def test_observe_can_read_its_own_event_from_store():
    store = InMemoryEpisodeStore()
    seen = []

    class Probe:
        def __init__(self, reader):
            self.reader = reader

        def observe(self, event):
            seen.extend(self.reader.get([event.id]))

        def handoff(self, query, ctx, k=6):
            return []

    feed(store, Probe(store), ev("e1", "cloud sql timeout"))
    assert [e.id for e in seen] == ["e1"]


def test_promote_all_sources_resolve_through_reader():
    store = InMemoryEpisodeStore()
    engine = PromoteAllEngine(store)
    feed(store, engine, ev("e1", "cloud sql needs a proxy"), ev("e2", "redis is fine"))

    handoffs = engine.handoff("cloud sql proxy", ctx())

    assert [h.id for h in handoffs] == ["e1"]
    assert all(isinstance(h, Handoff) for h in handoffs)
    for h in handoffs:
        assert [e.id for e in store.get(h.sources)] == list(h.sources)


def test_promote_all_hands_off_records_only():
    store = InMemoryEpisodeStore()
    engine = PromoteAllEngine(store)
    feed(
        store,
        engine,
        ev("e1", "cloud sql needs a proxy"),
        ev("r1", "cloud sql proxy", kind="recall", refs=("e1",)),
    )

    assert [h.id for h in engine.handoff("cloud sql proxy", ctx())] == ["e1"]


def test_no_memory_hands_off_nothing():
    store = InMemoryEpisodeStore()
    engine = NoMemoryEngine(store)
    feed(store, engine, ev("e1", "cloud sql needs a proxy"))

    assert engine.handoff("cloud sql", ctx()) == []
    assert engine.snapshot(T0) == []


def test_handoff_is_deterministic_for_same_event_sequence():
    def run():
        store = InMemoryEpisodeStore()
        engine = PromoteAllEngine(store)
        feed(store, engine, *(ev(f"e{i}", "deploy uses tags") for i in range(5)))
        return [h.id for h in engine.handoff("deploy tags", ctx(), k=3)]

    assert run() == run() == ["e0", "e1", "e2"]


def test_promote_all_snapshot_respects_time():
    store = InMemoryEpisodeStore()
    engine = PromoteAllEngine(store)
    feed(store, engine, ev("e1", "a", at=T0), ev("e2", "b", at=T0 + timedelta(days=1)))

    assert [h.id for h in engine.snapshot(T0)] == ["e1"]
