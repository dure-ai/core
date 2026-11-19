from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from cortex_mem.contract import Demote, EngineContext, Promote
from cortex_mem.engines.ledger import LedgerEngine
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.decide import canonical_of, decisions, knowledge_item
from cortex_mem.engines.ledger.model import Claim, Evidence
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.store.memory import InMemoryDatabase
from tests.ledger.fakes import FakeSimilarity
from tests.ledger.helpers import (
    T0,
    count,
    edit_state,
    ev,
    knowledge_ids,
    runtime,
    state,
)
from tests.ledger.test_store import FakeState

CFG = LedgerConfig.from_mapping(None)
FACT = "the deploy script must run migrations before restarting the api server every time"
# Jaccard with FACT is 11/13: the same claim on the lexical path, not a copy.
NEAR = "the deploy script must run migrations before restarting the api server each time"
OTHER = "always run database migrations first, then restart the api"


def both(*ids: str) -> pytest.MarkDecorator:
    """Run a test without a similarity port and with every id in one group."""
    return pytest.mark.parametrize(
        "groups", [None, dict.fromkeys(ids, "g")], ids=["lexical", "groups"]
    )


def promotes(rt) -> list[Promote]:
    return [v.decision for v in rt.verdicts if isinstance(v.decision, Promote)]


@both("e1")
def test_one_record_becomes_a_candidate_claim_and_is_not_promoted(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))

    assert knowledge_ids(db) == []
    st = state(db)
    claim = st.claim("c-e1")
    assert claim is not None and claim.status == "candidate"
    assert claim.canonical == "e1" and claim.scope == "team" and claim.owner == "t1"
    assert count(db, "mem:") == 1 and count(db, "claim:") == 1
    [evidence] = st.evidence("c-e1")
    assert (evidence.event_id, evidence.stance, evidence.derived) == ("e1", "support", None)
    assert (evidence.channel, evidence.key) == ("observation", "u:u1")
    memory = st.memory("e1")
    assert memory is not None and memory.claim_id == "c-e1"
    if groups:
        assert st.claims_in_group("e1") == ["c-e1"]
        assert memory.group == "e1"
    else:
        assert memory.group is None
        assert count(db, "grp:") == 0


@both("e1", "e2", "e3", "e4", "e5")
def test_one_user_repeating_does_not_promote(groups):
    rt, db = runtime(groups=groups)
    for i in range(1, 6):
        rt.append(ev(f"e{i}", FACT, session=f"s{i}", minutes=i))

    st = state(db)
    assert [c.id for c in st.claims()] == ["c-e1"]
    evidence = st.evidence("c-e1")
    assert len(evidence) == 5
    assert {e.key for e in evidence if e.derived is None} == {"u:u1"}
    assert st.claim("c-e1").status == "candidate"
    assert knowledge_ids(db) == [] and promotes(rt) == []


def test_two_people_saying_it_differently_in_one_group_promote():
    rt, db = runtime(groups={"e1": "g", "e2": "g"})
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))

    st = state(db)
    assert [c.id for c in st.claims()] == ["c-e1"]
    assert {e.key for e in st.evidence("c-e1")} == {"u:u1", "u:u2"}
    assert st.claim("c-e1").status == "established"
    assert st.claim("c-e1").promoted is True
    assert knowledge_ids(db) == ["c-e1"] and len(promotes(rt)) == 1
    assert rt.rejections == []

    # Without a port the two wordings are two claims and nothing is promoted.
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))
    st = state(db)
    assert [c.id for c in st.claims()] == ["c-e1", "c-e2"]
    assert all(c.status == "candidate" for c in st.claims())
    assert knowledge_ids(db) == [] and promotes(rt) == []


@both("e1", "e2")
def test_a_second_author_on_an_existing_claim_writes_evidence(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))

    st = state(db)
    assert st.memory("e2").claim_id == "c-e1"
    rows = {e.event_id: e for e in st.evidence("c-e1")}
    assert set(rows) == {"e1", "e2"}
    assert (rows["e2"].stance, rows["e2"].key, rows["e2"].derived) == ("support", "u:u2", None)


def test_a_tested_record_is_promoted():
    rt, db = runtime()
    rt.append(ev("e1", FACT, signal="test"))

    [promote] = promotes(rt)
    item = promote.item
    assert item.id == "c-e1" and item.kind == "knowledge"
    assert item.sources == ("e1",)
    assert item.body == FACT and item.title == FACT[:80]
    assert (item.scope, item.owner) == ("team", "t1")
    assert knowledge_ids(db) == ["c-e1"] and rt.rejections == []
    claim = state(db).claim("c-e1")
    assert claim.status == "established" and claim.promoted is True
    assert claim.report["decision"] == "established"
    assert claim.report["checks"]["sources"] is True


@both("e1", "e2")
def test_a_verbatim_copy_by_another_user_is_not_a_second_source(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", FACT, user="u2", minutes=1))

    st = state(db)
    rows = {e.event_id: e for e in st.evidence("c-e1")}
    assert rows["e2"].derived == "copy"
    assert st.claim("c-e1").status == "candidate"
    assert knowledge_ids(db) == [] and promotes(rt) == []


@both("e1", "e2")
def test_a_record_that_cites_what_it_was_handed_is_not_a_vote(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", NEAR, user="u2", refs=("c-e1",), minutes=1))

    st = state(db)
    rows = {e.event_id: e for e in st.evidence("c-e1")}
    assert rows["e2"].derived == "retrieved"
    assert st.claim("c-e1").status == "candidate"
    assert knowledge_ids(db) == []


@both("e1", "e2")
def test_a_record_after_its_session_was_handed_the_claim_is_retrieved(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    edit_state(db, lambda st: st.mark_seen("s2", "c-e1", T0 + timedelta(minutes=1)))
    rt.append(ev("e2", NEAR, user="u2", session="s2", minutes=2))

    rows = {e.event_id: e for e in state(db).evidence("c-e1")}
    assert rows["e2"].derived == "retrieved"
    assert knowledge_ids(db) == []


@both("e1", "e2")
def test_a_record_before_its_session_was_handed_the_claim_is_a_vote(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    # Seen at the very time of the record: not earlier, so not retrieved.
    edit_state(db, lambda st: st.mark_seen("s2", "c-e1", T0 + timedelta(minutes=2)))
    rt.append(ev("e2", NEAR, user="u2", session="s2", minutes=2))

    rows = {e.event_id: e for e in state(db).evidence("c-e1")}
    assert rows["e2"].derived is None
    assert knowledge_ids(db) == ["c-e1"]


@both("e1", "e2")
def test_a_new_record_does_not_join_a_deprecated_claim(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))

    def deprecate(st: LedgerStore) -> None:
        claim = st.claim("c-e1")
        claim.status = "deprecated"
        st.put_claim(claim)

    edit_state(db, deprecate)
    rt.append(ev("e2", FACT, user="u2", minutes=1))

    st = state(db)
    assert st.memory("e2").claim_id == "c-e2"
    assert st.claim("c-e2").status == "candidate"
    assert [e.event_id for e in st.evidence("c-e2")] == ["e2"]
    assert [e.event_id for e in st.evidence("c-e1")] == ["e1"]
    assert st.claim("c-e1").status == "deprecated"
    if groups:
        assert st.claims_in_group("e1") == ["c-e1", "c-e2"]


@both("e1", "e2")
def test_personal_records_use_the_session_as_the_source(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT, scope="personal", session="s1"))
    rt.append(ev("e2", FACT, scope="personal", session="s2", minutes=1))

    st = state(db)
    assert {e.key for e in st.evidence("c-e1")} == {"s:s1", "s:s2"}
    [promote] = promotes(rt)
    assert promote.item.scope == "personal" and promote.item.owner == "u1"
    assert knowledge_ids(db) == ["c-e1"] and rt.rejections == []


def test_personal_records_in_one_session_do_not_promote():
    rt, db = runtime()
    rt.append(ev("e1", FACT, scope="personal", session="s1"))
    rt.append(ev("e2", FACT, scope="personal", session="s1", minutes=1))
    assert knowledge_ids(db) == []


@both("e1", "e2")
def test_redelivery_does_not_count_twice(groups):
    """Review Focus 1: observe may run twice for one event on one state."""
    db = InMemoryDatabase()
    engine = LedgerEngine()
    st = FakeState()
    e1 = ev("e1", FACT)
    e2 = ev("e2", OTHER if groups else NEAR, user="u2", minutes=1)

    def ctx(upto: int) -> EngineContext:
        similar = None if groups is None else FakeSimilarity(db.events, groups, upto=upto)
        return EngineContext(db.events, db.knowledge, st, similar)

    db.events.append(e1)
    first = engine.observe(e1, ctx(1))
    again = engine.observe(e1, ctx(1))
    assert first == [] and again == []
    db.events.append(e2)
    first = engine.observe(e2, ctx(2))
    again = engine.observe(e2, ctx(2))
    assert [type(d) for d in first] == [Promote]
    assert again == []

    store = LedgerStore(st)
    assert [c.id for c in store.claims()] == ["c-e1"]
    assert [e.event_id for e in store.evidence("c-e1")] == ["e1", "e2"]
    grp = [k for k in st.data if k.startswith("grp:")]
    assert len(grp) == (1 if groups else 0)
    assert store.skipped() == {}


@both("e1", "e2", "e3")
def test_knowledge_sources_exclude_derived_evidence(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", FACT, user="u3", minutes=1))  # a copy of e1
    rt.append(ev("e3", NEAR, user="u2", minutes=2))

    st = state(db)
    rows = {e.event_id: e.derived for e in st.evidence("c-e1")}
    assert rows == {"e1": None, "e2": "copy", "e3": None}
    [promote] = promotes(rt)
    assert promote.item.sources == ("e1", "e3")
    [item] = db.knowledge.all()
    assert item.sources == ("e1", "e3")


def test_a_stronger_record_on_established_knowledge_replaces_its_body():
    rt, db = runtime(groups={"e1": "g", "e2": "g", "e3": "g"})
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))
    tested = "migrations run before the api restarts; checked by the deploy test"
    rt.append(ev("e3", tested, user="u3", signal="test", minutes=2))

    first, second = promotes(rt)
    assert first.item.body == OTHER  # equal strength: the later record
    assert second.item.body == tested
    assert second.item.sources == ("e1", "e2", "e3")
    assert state(db).claim("c-e1").canonical == "e3"
    [item] = db.knowledge.all()
    assert item.body == tested


def test_a_derived_record_on_established_knowledge_decides_nothing():
    rt, db = runtime(groups={"e1": "g", "e2": "g", "e3": "g"})
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))
    rt.append(ev("e3", OTHER, user="u3", minutes=2))  # a copy of e2
    assert len(promotes(rt)) == 1
    assert state(db).claim("c-e1").canonical == "e2"


def test_records_with_other_conditions_are_other_claims():
    rt, db = runtime(groups={"e1": "g", "e2": "g"})
    rt.append(ev("e1", FACT, conditions={"os": "mac"}))
    rt.append(ev("e2", OTHER, user="u2", conditions={"os": "linux"}, minutes=1))
    st = state(db)
    assert [c.id for c in st.claims()] == ["c-e1", "c-e2"]
    assert st.claim("c-e2").conditions == {"os": "linux"}
    assert knowledge_ids(db) == []


@both("e1")
def test_an_org_record_is_counted_and_left_alone(groups):
    rt, db = runtime(groups=groups)
    rt.append(ev("e1", FACT, scope="org", signal="test"))

    st = state(db)
    assert st.claims() == [] and count(db, "mem:") == 0 and count(db, "grp:") == 0
    assert count(db, "sig:") == 0
    assert st.skipped() == {"org_scope": 1}
    assert knowledge_ids(db) == []


def test_tick_and_handoff_do_nothing_yet():
    rt, db = runtime()
    rt.append(ev("e1", FACT, signal="test"))
    engine = rt.engine
    ctx = EngineContext(db.events, db.knowledge, FakeState())
    assert engine.tick(T0, ctx) == []
    assert engine.handoff("deploy", None, "team", ctx) == []  # type: ignore[arg-type]


# --- decide.py ---------------------------------------------------------------


def _claim(**kw) -> Claim:
    base = Claim(
        id="c-e1", scope="team", owner="t1", conditions={}, canonical="e1",
        status="candidate", first_at=T0, last_support_at=T0, invalid_since=None,
        promoted=False, use_success=0, use_failure=0, report={"S": 1.0, "C": 2.0},
    )
    return replace(base, **kw)


def _evidence(eid, *, channel="observation", minutes=0, derived=None, stance="support",
              user="u1"):
    return Evidence(
        claim_id="c-e1", event_id=eid, stance=stance, channel=channel, key=f"u:{user}",
        user=user, at=T0 + timedelta(minutes=minutes), derived=derived,
    )


def test_canonical_prefers_the_strongest_then_the_latest_then_the_largest_id():
    rows = [
        _evidence("e1", channel="test", minutes=0),
        _evidence("e2", channel="observation", minutes=5),
        _evidence("e3", channel="doc", minutes=0),
    ]
    assert canonical_of(rows, CFG) == "e3"  # test and doc tie; same time; larger id
    rows.append(_evidence("e4", channel="doc", minutes=1))
    assert canonical_of(rows, CFG) == "e4"  # later
    rows.append(_evidence("e5", channel="human", minutes=0, derived="copy"))
    rows.append(_evidence("e6", channel="human", minutes=0, stance="contradict"))
    assert canonical_of(rows, CFG) == "e4"  # derived and contradicting rows are not candidates


def test_canonical_skips_judgement_evidence_and_falls_back_to_the_first_memory():
    rows = [
        _evidence("e1", minutes=0, derived="copy"),
        _evidence("o1", channel="human", minutes=1),
        _evidence("e2", minutes=2, derived="retrieved"),
    ]
    records = {"e1", "e2"}
    assert canonical_of(rows, CFG, records=records) == "e1"
    rows.append(_evidence("e3", minutes=3))
    assert canonical_of(rows, CFG, records={"e1", "e2", "e3"}) == "e3"


def test_knowledge_item_takes_the_first_line_as_title_and_skips_derived_sources():
    claim = _claim(conditions={"os": "mac"})
    rows = [
        _evidence("e2", minutes=1),
        _evidence("e9", minutes=0, derived="retrieved"),
        _evidence("e1", minutes=0),
        _evidence("x1", minutes=3, stance="contradict"),
    ]
    rows.sort(key=lambda e: (e.at, e.event_id))
    text = "  " + "a" * 100 + "\nsecond line"
    item = knowledge_item(claim, rows, text)
    assert item.title == "a" * 80
    assert item.body == text
    assert item.sources == ("e1", "e2")
    assert dict(item.conditions) == {"os": "mac"}


def test_a_promoted_claim_that_becomes_deprecated_is_demoted():
    trigger = ev("o1", "", kind="outcome", signal="test", outcome="failure", minutes=9)
    claim = _claim(status="deprecated", promoted=True)
    out = decisions("established", "e1", claim, [_evidence("e1")], "x", trigger)
    assert len(out) == 1 and isinstance(out[0], Demote)
    demote = out[0]
    assert (demote.id, demote.sources, demote.why) == ("c-e1", ("o1",), "invalid")
    assert demote.since == trigger.at and demote.reason
    assert claim.promoted is False and claim.invalid_since == trigger.at


def test_an_unpromoted_claim_that_becomes_deprecated_decides_nothing():
    trigger = ev("o1", "", kind="outcome", signal="test", outcome="failure", minutes=9)
    claim = _claim(status="deprecated", promoted=False)
    assert decisions("candidate", "e1", claim, [_evidence("e1")], "x", trigger) == []
    assert claim.invalid_since == trigger.at


def test_established_with_the_same_canonical_decides_nothing():
    trigger = ev("e2", "x", minutes=1)
    claim = _claim(status="established", promoted=True)
    assert decisions("established", "e1", claim, [_evidence("e1")], "x", trigger) == []
    claim = _claim(status="disputed", promoted=True)
    assert decisions("established", "e1", claim, [_evidence("e1")], "x", trigger) == []
