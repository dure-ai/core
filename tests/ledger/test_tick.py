from __future__ import annotations

from datetime import timedelta

from cortex_mem.contract import Caller, Demote
from cortex_mem.engines.ledger import LedgerEngine
from tests.ledger.helpers import T0, ev, knowledge_ids, runtime, state

FACT = "the deploy script must run migrations before restarting the api server every time"
NEAR = "the deploy script must run migrations before restarting the api server each time"
ALPHA = "service alpha rotates signing keys quarterly and publishes new keys to every region"
ALPHA_NEAR = "service alpha rotates signing keys quarterly and publishes new keys to each region"
BETA = "service beta invalidates cache entries after each schema change to avoid stale reads"
BETA_NEAR = "service beta invalidates cache entries after every schema change to avoid stale reads"
GAMMA = "service gamma drains queue workers gracefully before any node reboot happens"
GAMMA_NEAR = "service gamma drains queue workers gracefully before any node reboot occurs"

REVIEW_90 = {"review_after_days": "90"}


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
