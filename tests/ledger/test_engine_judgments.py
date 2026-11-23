from __future__ import annotations

from datetime import timedelta

from cortex_mem.contract import Caller, Demote, EngineContext, Promote
from cortex_mem.engines.ledger import LedgerEngine
from cortex_mem.engines.ledger.store import LedgerStore

from .helpers import (
    ENGINE,
    T0,
    UNVERIFIED_TEAM,
    _Writable,
    edit_state,
    ev,
    knowledge_ids,
    runtime,
    state,
)

FACT = "the deploy script must run migrations before restarting the api server every time"
# Jaccard with FACT is 11/13: the same claim on the lexical path, not a copy.
NEAR = "the deploy script must run migrations before restarting the api server each time"


def decisions_of(rt, kind):
    return [v.decision for v in rt.verdicts if isinstance(v.decision, kind)]


def judgment(id, user, *, kind="outcome", outcome="success", signal="review", refs=("e1",),
             minutes=1, scope="team", session=None):
    return ev(id, "", kind=kind, user=user, signal=signal, outcome=outcome, refs=refs,
              minutes=minutes, scope=scope, session=session)


def two_observers(rt):
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))


def test_two_people_confirming_by_reference_promote():
    rt, db = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(judgment("e2", "u2", minutes=1))
    rt.append(judgment("e3", "u3", minutes=2))

    assert knowledge_ids(db) == ["c-e1"]
    [promote] = decisions_of(rt, Promote)
    assert {"e1", "e2"} <= set(promote.item.sources)
    assert state(db).claim("c-e1").status == "established"
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1", "e2", "e3"}


def test_a_personal_confirmation_is_not_evidence_for_team_knowledge():
    rt, db = runtime(groups={"e1": "g", "e3": "g"})
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(judgment("e2", "u2", kind="human_confirm", signal=None, scope="personal", minutes=1))
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1"}
    assert state(db).skipped()["out_of_scope_evidence"] == 1

    rt.append(ev("e3", NEAR, user="u3", minutes=2))
    assert knowledge_ids(db) == ["c-e1"]
    assert rt.rejections == []
    [promote] = decisions_of(rt, Promote)
    assert "e2" not in promote.item.sources
    assert "e2" not in {e.event_id for e in state(db).evidence("c-e1")}


def test_a_human_confirmation_promotes_alone():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", kind="human_confirm", signal=None, minutes=1))
    assert knowledge_ids(db) == ["c-e1"]
    assert state(db).evidence("c-e1")[-1].channel == "human"


def test_an_agents_own_success_report_is_usefulness_not_truth():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(ev("r1", "", kind="recall", user="u2", session="s9", refs=("c-e1",), minutes=1))
    rt.append(judgment("e2", "u2", signal="self", session="s9", minutes=2))

    st = state(db)
    assert {e.event_id for e in st.evidence("c-e1")} == {"e1"}
    claim = st.claim("c-e1")
    assert claim.use_success == 1 and claim.status == "candidate"
    assert knowledge_ids(db) == []


def test_a_success_report_from_a_session_never_handed_the_claim_counts_nothing():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", signal=None, minutes=1))
    assert state(db).claim("c-e1").use_success == 0


def test_a_success_report_dated_before_the_session_was_handed_the_claim_counts_nothing():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(ev("r1", "", kind="recall", user="u2", session="s9", refs=("c-e1",), minutes=5))
    rt.append(judgment("e2", "u2", signal="self", session="s9", minutes=2))
    assert state(db).claim("c-e1").use_success == 0

    rt.append(judgment("e3", "u2", signal="self", session="s9", minutes=5))
    assert state(db).claim("c-e1").use_success == 1  # handed at the same time: counts


def test_an_agents_own_failure_report_does_not_demote_alone():
    rt, db = runtime()
    two_observers(rt)
    assert knowledge_ids(db) == ["c-e1"]
    rt.append(judgment("e3", "u3", signal="self", outcome="failure", minutes=2))

    claim = state(db).claim("c-e1")
    assert knowledge_ids(db) == ["c-e1"]
    assert claim.status in ("established", "disputed")
    assert claim.use_failure == 1
    [row] = [e for e in state(db).evidence("c-e1") if e.event_id == "e3"]
    assert (row.stance, row.channel) == ("contradict", "observation")


def test_a_tested_failure_demotes_with_the_reason():
    rt, db = runtime()
    two_observers(rt)
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=5))

    [demote] = decisions_of(rt, Demote)
    assert demote.why == "invalid" and demote.since == T0 + timedelta(minutes=5)
    assert demote.sources == ("e3",)
    assert knowledge_ids(db) == []
    claim = state(db).claim("c-e1")
    assert claim.status == "deprecated" and claim.invalid_since == demote.since


def test_a_demote_cites_the_verified_failures_not_only_the_one_that_tipped_it():
    rt, db = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(judgment("e2", "u2", signal="review", minutes=1))
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=2))
    assert state(db).claim("c-e1").status == "disputed"

    rt.append(judgment("e4", "u4", signal="self", outcome="failure", minutes=3))

    [demote] = decisions_of(rt, Demote)
    assert demote.why == "invalid" and demote.since == T0 + timedelta(minutes=3)
    assert demote.sources == ("e3", "e4")
    assert rt.rejections == []


def test_a_human_confirmed_claim_survives_one_test_failure():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", kind="human_confirm", signal=None, minutes=1))
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=2))
    assert knowledge_ids(db) == ["c-e1"]
    assert decisions_of(rt, Demote) == []
    assert state(db).claim("c-e1").status != "deprecated"


def org_judgment(id, user, team, *, signal, outcome, minutes):
    return ev(id, "", kind="outcome", user=user, team=team, signal=signal, outcome=outcome,
              refs=("e1",), scope="org", minutes=minutes)


def team_handoff(rt):
    caller = Caller(user="u3", team="t1", session="s9", at=T0 + timedelta(minutes=10))
    return [(h.id, h.status) for h in rt.handoff("deploy migrations api server", caller, "team")]


def test_an_org_success_from_another_team_is_not_evidence():
    rt, db = runtime()
    two_observers(rt)
    assert team_handoff(rt) == [("c-e1", "knowledge")]

    rt.append(org_judgment("e3", "u9", "t2", signal="review", outcome="success", minutes=2))

    assert team_handoff(rt) == [("c-e1", "knowledge")]
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1", "e2"}
    assert state(db).skipped() == {"org_scope": 1}
    assert rt.rejections == []


def test_an_org_test_failure_from_another_team_does_not_demote_team_knowledge():
    rt, db = runtime()
    two_observers(rt)

    rt.append(org_judgment("e3", "z1", "t2", signal="test", outcome="failure", minutes=2))

    assert decisions_of(rt, Demote) == []
    assert knowledge_ids(db) == ["c-e1"]
    assert state(db).claim("c-e1").status == "established"
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1", "e2"}
    assert state(db).skipped() == {"org_scope": 1}


def test_judgment_refs_may_name_the_claim_or_its_record():
    results = []
    for ref in ("c-e1", "e1"):
        rt, db = runtime()
        rt.append(ev("e1", FACT))
        rt.append(judgment("e2", "u2", refs=(ref,), minutes=1))
        claim = state(db).claim("c-e1")
        results.append((claim.status, [(e.event_id, e.stance) for e in state(db).evidence("c-e1")]))
    assert results[0] == results[1]
    assert results[0][0] == "established"


def test_unknown_refs_are_counted_not_raised():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", refs=("nope", "e1"), minutes=1))
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1", "e2"}
    assert state(db).skipped() == {"unknown_ref": 1}


def test_an_unknown_outcome_is_counted_and_writes_nothing():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", outcome="unknown", minutes=1))
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1"}
    assert state(db).skipped() == {"unknown_outcome": 1}


def test_a_deprecated_claim_revives_on_later_verified_support():
    rt, db = runtime()
    two_observers(rt)
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=5))
    assert state(db).claim("c-e1").status == "deprecated"

    rt.append(judgment("e4", "u4", kind="human_confirm", signal=None, minutes=10))
    claim = state(db).claim("c-e1")
    assert claim.status != "deprecated" and claim.invalid_since is None


def test_a_deprecated_claim_ignores_support_that_is_not_verified_or_not_later():
    rt, db = runtime()
    two_observers(rt)
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=5))
    rt.append(judgment("e4", "u4", signal="review", minutes=3))  # earlier than the failure
    rt.append(judgment("e5", "u5", signal="self", outcome="success", minutes=9))
    rt.append(judgment("e6", "u6", signal="self", outcome="success", minutes=10, refs=("c-e1",)))

    claim = state(db).claim("c-e1")
    assert claim.status == "deprecated" and claim.invalid_since is not None
    assert knowledge_ids(db) == []
    assert len(decisions_of(rt, Promote)) == 1  # only the first promotion


def test_a_promoted_claim_keeps_its_status_when_the_checks_would_fail():
    rt, db = runtime(engine=LedgerEngine(UNVERIFIED_TEAM))
    two_observers(rt)
    assert state(db).claim("c-e1").status == "established"
    # a canonical with a high-risk term: re-evaluation alone must not demote
    rt2, db2 = runtime(engine=LedgerEngine({**UNVERIFIED_TEAM, "high_risk_terms": "migrations"}))
    rt2.append(ev("e1", FACT, user="u1"))
    rt2.append(ev("e2", NEAR, user="u2", minutes=1))
    assert state(db2).claim("c-e1").status != "established"  # not promoted, high risk
    edit_state(db2, lambda st: _promote(st))
    rt2.append(judgment("e3", "u3", signal="review", minutes=2))
    assert state(db2).claim("c-e1").status == "established"


def _promote(st):
    claim = st.claim("c-e1")
    claim.status = "established"
    claim.promoted = True
    st.put_claim(claim)


def test_a_judgment_observed_twice_counts_once():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    rt.append(ev("r1", "", kind="recall", user="u2", session="s9", refs=("c-e1",), minutes=1))
    success = judgment("e2", "u2", signal="self", session="s9", minutes=2)
    failure = judgment("e3", "u3", signal="self", outcome="failure", minutes=3)
    rt.append(success)
    rt.append(failure)

    staged = _Writable(db.state(ENGINE))  # committed state plus this run's writes
    ctx = EngineContext(events=rt.events, knowledge=rt.knowledge, state=staged, similar=None)
    engine = LedgerEngine(UNVERIFIED_TEAM)
    for _ in range(2):
        engine.observe(success, ctx)
        engine.observe(failure, ctx)

    claim = LedgerStore(staged).claim("c-e1")
    assert (claim.use_success, claim.use_failure) == (1, 1)
    assert [e.event_id for e in LedgerStore(staged).evidence("c-e1")].count("e3") == 1


def test_recall_marks_what_the_session_was_handed():
    rt, db = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    caller = Caller(user="u2", team="t1", session="s9", at=ev("x", "", minutes=1).at)
    [handed] = rt.recall("deploy migrations api server", caller, "team", "r1")
    assert handed.id == "c-e1"
    assert state(db).seen("s9", "c-e1") == caller.at

    rt.append(ev("e2", NEAR, user="u2", session="s9", minutes=2))
    rows = {e.event_id: e.derived for e in state(db).evidence("c-e1")}
    assert rows["e2"] == "retrieved"


RISKY = "always run database migrations first, then restart the api, using the admin password"
RISK_CFG = {**UNVERIFIED_TEAM, "high_risk_terms": "password"}


def risky_setup():
    rt, db = runtime(
        engine=LedgerEngine(RISK_CFG), groups={"e1": "g", "e2": "g", "e3": "g", "e5": "g"}
    )
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))
    assert state(db).claim("c-e1").status == "established"
    return rt, db


def test_a_stronger_high_risk_record_cannot_replace_the_body_without_human_support():
    rt, db = risky_setup()
    rt.append(ev("e3", RISKY, user="u3", signal="test", minutes=2))

    assert len(decisions_of(rt, Promote)) == 1
    assert state(db).claim("c-e1").canonical == "e1"
    [item] = db.knowledge.all()
    assert "password" not in item.body


def test_a_human_confirmed_high_risk_record_replaces_the_body():
    rt, db = risky_setup()
    rt.append(ev("e3", RISKY, user="u3", signal="human", minutes=2))

    assert len(decisions_of(rt, Promote)) == 2
    assert state(db).claim("c-e1").canonical == "e3"
    [item] = db.knowledge.all()
    assert "password" in item.body


def test_a_sticky_established_claim_without_a_canonical_change_does_not_promote_again():
    rt, db = risky_setup()
    rt.append(judgment("e3", "u3", signal="review", minutes=2))
    assert state(db).claim("c-e1").status == "established"
    assert len(decisions_of(rt, Promote)) == 1


def test_human_support_given_before_a_riskier_canonical_does_not_cover_it():
    rt, db = runtime(
        engine=LedgerEngine(RISK_CFG), groups={"e1": "g", "e3": "g", "e4": "g"}
    )
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(judgment("e2", "u2", kind="human_confirm", signal=None, minutes=1))
    assert len(decisions_of(rt, Promote)) == 1

    rt.append(ev("e3", RISKY, user="u3", signal="test", minutes=2))
    assert len(decisions_of(rt, Promote)) == 1
    assert state(db).claim("c-e1").canonical == "e1"
    assert "password" not in db.knowledge.all()[0].body

    rt.append(judgment("e4", "u4", kind="human_confirm", signal=None, minutes=3))
    assert len(decisions_of(rt, Promote)) == 2
    assert state(db).claim("c-e1").canonical == "e3"
    assert "password" in db.knowledge.all()[0].body


def test_a_disputed_claim_does_not_take_a_high_risk_canonical_either():
    rt, db = risky_setup()
    rt.append(judgment("e3", "u3", signal="self", outcome="failure", minutes=2))
    rt.append(judgment("e4", "u4", signal="self", outcome="failure", minutes=3))
    assert state(db).claim("c-e1").status == "disputed"
    rt.append(ev("e5", RISKY, user="u5", signal="test", minutes=4))
    assert state(db).claim("c-e1").canonical == "e1"
    assert len(decisions_of(rt, Promote)) == 1


def test_skipped_events_are_counted_once_when_observed_again():
    rt, db = runtime()
    rt.append(ev("e1", FACT))
    events = [
        ev("o1", FACT, scope="org", minutes=1),
        judgment("j1", "u2", refs=("nope", "e1"), minutes=2),
        judgment("j2", "u2", outcome="unknown", minutes=3),
        org_judgment("j3", "u9", "t2", signal="test", outcome="failure", minutes=4),
        judgment("j4", "u2", kind="human_confirm", signal=None, scope="personal", minutes=5),
    ]
    for event in events:
        rt.append(event)
    expected = {"org_scope": 2, "unknown_ref": 1, "unknown_outcome": 1,
                "out_of_scope_evidence": 1}
    assert state(db).skipped() == expected

    staged = _Writable(db.state(ENGINE))
    ctx = EngineContext(events=rt.events, knowledge=rt.knowledge, state=staged, similar=None)
    engine = LedgerEngine(UNVERIFIED_TEAM)
    for event in events:
        engine.observe(event, ctx)

    assert LedgerStore(staged).skipped() == expected
