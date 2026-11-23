from __future__ import annotations

from datetime import timedelta

from cortex_mem.contract import Caller, Demote, Promote
from cortex_mem.engines.ledger import LedgerEngine

from .helpers import T0, UNVERIFIED_TEAM, ev, knowledge_ids, runtime, state

FACT = "the deploy script must run migrations before restarting the api server every time"
NEAR = "the deploy script must run migrations before restarting the api server each time"
ALPHA = "service alpha rotates signing keys quarterly and publishes new keys to every region"
ALPHA_NEAR = "service alpha rotates signing keys quarterly and publishes new keys to each region"
BETA = "service beta invalidates cache entries after each schema change to avoid stale reads"
BETA_NEAR = "service beta invalidates cache entries after every schema change to avoid stale reads"
GAMMA = "service gamma drains queue workers gracefully before any node reboot happens"
GAMMA_NEAR = "service gamma drains queue workers gracefully before any node reboot occurs"

REVIEW_90 = {**UNVERIFIED_TEAM, "review_after_days": "90"}


def decisions_of(rt, kind):
    return [v.decision for v in rt.verdicts if isinstance(v.decision, kind)]


def caller(at):
    return Caller(user="u1", team="t1", session="sx", at=at)


def two_observers(rt):
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))


def test_tick_does_nothing_by_default():
    rt, db = runtime()
    two_observers(rt)

    rt.tick(T0 + timedelta(days=365))

    assert decisions_of(rt, Demote) == []
    assert knowledge_ids(db) == ["c-e1"]
    assert state(db).claim("c-e1").status == "established"


def test_tick_archives_established_claims_without_recent_support():
    rt, db = runtime(LedgerEngine(REVIEW_90))
    two_observers(rt)

    rt.tick(T0 + timedelta(days=80))
    assert decisions_of(rt, Demote) == []
    assert knowledge_ids(db) == ["c-e1"]
    assert state(db).claim("c-e1").status == "established"

    now = T0 + timedelta(days=100)
    rt.tick(now)
    [demote] = decisions_of(rt, Demote)
    assert demote.id == "c-e1"
    assert demote.why == "stale"
    assert demote.sources == ("e1",)
    assert demote.since == now
    assert demote.reason
    assert knowledge_ids(db) == []
    claim = state(db).claim("c-e1")
    assert claim.status == "archived"
    assert claim.promoted is False

    verdicts = len(rt.verdicts)
    rt.tick(now + timedelta(days=1))
    assert len(rt.verdicts) == verdicts  # idempotent: nothing more to decide
    assert state(db).claim("c-e1").status == "archived"


def test_archived_claims_take_no_new_records_and_are_not_handed_off():
    rt, db = runtime(LedgerEngine(REVIEW_90))
    two_observers(rt)
    now = T0 + timedelta(days=100)
    rt.tick(now)
    assert knowledge_ids(db) == []

    assert rt.handoff("deploy migrations api server", caller(now), "team") == []

    rt.append(ev("e3", FACT, user="u1", minutes=60 * 24 * 101))

    assert state(db).claim("c-e1").status == "archived"
    fresh = state(db).claim("c-e3")
    assert fresh is not None and fresh.status == "candidate"
    assert knowledge_ids(db) == []


DAY = 60 * 24  # minutes
ARCHIVED_AT = T0 + timedelta(days=100)


def judgment(id, user, *, signal, outcome, minutes, kind="outcome"):
    return ev(id, "", kind=kind, user=user, signal=signal, outcome=outcome, refs=("e1",),
              minutes=minutes)


def archived_runtime():
    rt, db = runtime(LedgerEngine(REVIEW_90))
    two_observers(rt)
    rt.tick(ARCHIVED_AT)
    assert state(db).claim("c-e1").status == "archived"
    return rt, db


def test_a_tick_records_when_a_claim_was_archived():
    _, db = archived_runtime()
    assert state(db).claim("c-e1").invalid_since == ARCHIVED_AT


def test_failures_on_an_archived_claim_collect_evidence_but_do_not_revive_it():
    rt, db = archived_runtime()
    rt.append(judgment("e3", "u3", signal="self", outcome="failure", minutes=101 * DAY))
    rt.append(judgment("e4", "u4", signal="test", outcome="failure", minutes=102 * DAY))

    claim = state(db).claim("c-e1")
    assert claim.status == "archived" and claim.use_failure == 1
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1", "e2", "e3", "e4"}
    assert rt.handoff("deploy migrations api server", caller(T0 + timedelta(days=103)), "team") == []
    assert knowledge_ids(db) == []
    assert len(decisions_of(rt, Demote)) == 1  # only the stale one

    rt.tick(T0 + timedelta(days=400))
    assert state(db).claim("c-e1").status == "archived"


def test_support_that_is_not_verified_or_not_after_the_archival_does_not_revive():
    rt, db = archived_runtime()
    rt.append(judgment("e3", "u3", signal="review", outcome="success", minutes=50 * DAY))
    rt.append(ev("r1", "", kind="recall", user="u4", session="s9", refs=("c-e1",),
                 minutes=101 * DAY))
    rt.append(ev("e4", "", kind="outcome", user="u4", session="s9", signal="self",
                 outcome="success", refs=("e1",), minutes=102 * DAY))

    claim = state(db).claim("c-e1")
    assert claim.status == "archived" and claim.invalid_since == ARCHIVED_AT
    assert knowledge_ids(db) == []
    assert len(decisions_of(rt, Promote)) == 1


def test_verified_support_after_the_archival_revives_and_promotes_again():
    rt, db = archived_runtime()
    rt.append(judgment("e3", "u3", signal="review", outcome="success", minutes=101 * DAY))

    claim = state(db).claim("c-e1")
    assert claim.status == "established" and claim.invalid_since is None
    assert knowledge_ids(db) == ["c-e1"]
    assert len(decisions_of(rt, Promote)) == 2
    assert rt.rejections == []


def test_a_revived_archived_claim_is_promoted_only_if_the_gate_passes():
    rt, db = archived_runtime()
    rt.append(judgment("e3", "u3", signal="self", outcome="failure", minutes=101 * DAY))
    rt.append(judgment("e4", "u4", signal="self", outcome="failure", minutes=101 * DAY + 1))
    rt.append(judgment("e5", "u5", signal="review", outcome="success", minutes=102 * DAY))

    claim = state(db).claim("c-e1")
    assert claim.status == "provisional" and claim.invalid_since is None
    assert claim.report["checks"]["contradiction"] is False
    assert knowledge_ids(db) == []
    assert len(decisions_of(rt, Promote)) == 1


def test_tick_order_is_stable_by_claim_id():
    rt, db = runtime(LedgerEngine(REVIEW_90))
    # Created in the order e2, e10, e3, but claim ids sort as c-e10, c-e2, c-e3.
    rt.append(ev("e2", ALPHA, user="u1", minutes=0))
    rt.append(ev("e2n", ALPHA_NEAR, user="u2", minutes=1))
    rt.append(ev("e10", BETA, user="u1", minutes=2))
    rt.append(ev("e10n", BETA_NEAR, user="u2", minutes=3))
    rt.append(ev("e3", GAMMA, user="u1", minutes=4))
    rt.append(ev("e3n", GAMMA_NEAR, user="u2", minutes=5))
    assert knowledge_ids(db) == ["c-e10", "c-e2", "c-e3"]

    rt.tick(T0 + timedelta(days=100))

    demotes = decisions_of(rt, Demote)
    assert [d.id for d in demotes] == ["c-e10", "c-e2", "c-e3"]
    assert all(d.why == "stale" for d in demotes)
    assert knowledge_ids(db) == []


def test_tick_leaves_claims_that_are_not_established_alone():
    rt, db = runtime(LedgerEngine(REVIEW_90))
    rt.append(ev("a1", ALPHA, user="u1"))  # candidate
    rt.append(ev("b1", BETA, user="u1", minutes=1))
    rt.append(ev("bx", "", kind="outcome", user="u3", signal="self", outcome="failure",
                 refs=("b1",), minutes=2))
    rt.append(ev("b2", BETA_NEAR, user="u2", minutes=3))  # provisional: too contested
    two_observers(rt)
    rt.append(judgment("e3", "u3", signal="self", outcome="failure", minutes=4))
    rt.append(judgment("e4", "u4", signal="self", outcome="failure", minutes=5))  # disputed
    rt.append(ev("g1", GAMMA, user="u1", minutes=6))
    rt.append(ev("g2", GAMMA_NEAR, user="u2", minutes=7))
    rt.append(ev("gx", "", kind="outcome", user="u3", signal="test", outcome="failure",
                 refs=("g1",), minutes=8))  # deprecated
    before = [c.to_json() for c in state(db).claims()]
    assert [(c["id"], c["status"]) for c in before] == [
        ("c-a1", "candidate"),
        ("c-b1", "provisional"),
        ("c-e1", "disputed"),
        ("c-g1", "deprecated"),
    ]
    verdicts = len(rt.verdicts)

    rt.tick(T0 + timedelta(days=400))

    assert len(rt.verdicts) == verdicts
    assert [c.to_json() for c in state(db).claims()] == before
    assert knowledge_ids(db) == ["c-e1"]  # disputed knowledge stays
