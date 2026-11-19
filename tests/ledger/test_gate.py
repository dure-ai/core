from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.evidence import Tally
from cortex_mem.engines.ledger.gate import GateReport, evaluate
from cortex_mem.engines.ledger.model import Claim

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 19, 10, 0, tzinfo=KST)
CFG = LedgerConfig.from_mapping(None)


def claim(status="candidate", scope="team") -> Claim:
    return Claim(id="c1", scope=scope, owner="t1", conditions={}, canonical="x",
                 status=status, first_at=T0, last_support_at=T0,
                 invalid_since=None, promoted=False, use_success=0, use_failure=0)


def tl(S=0.7, C=0.0, keys=1, verified=0, contra_keys=0, vc=False,
       span_days=0, by=None) -> Tally:
    by = by if by is not None else {"observation": keys}
    return Tally(S=S, C=C, support_keys=keys, verified_keys=verified,
                 contra_keys=contra_keys, verified_contra=vc,
                 first_support=T0 if keys else None,
                 last_support=T0 + timedelta(days=span_days) if keys else None,
                 by_channel=by, confidence=0.5)


def cfg(**gate) -> LedgerConfig:
    base = CFG.gates["team"]
    return replace(CFG, gates={**CFG.gates, "team": replace(base, **gate)})


def test_single_observation_is_candidate():
    r = evaluate(claim(), tl(), "text", CFG)
    assert r.decision == "candidate"
    assert r.checks["sources"] is False


def test_two_people_established():
    r = evaluate(claim(), tl(S=1.4, keys=2), "text", CFG)
    assert r.decision == "established"
    assert all(r.checks.values())


def test_one_person_with_test_established_by_alt_verified():
    r = evaluate(claim(), tl(S=2.0, keys=1, verified=1, by={"test": 1}), "text", CFG)
    assert r.decision == "established"


def test_require_verified_leaves_provisional():
    r = evaluate(claim(), tl(S=1.4, keys=2), "text", cfg(require_verified=True))
    assert r.decision == "provisional"
    assert r.checks["verified"] is False


def test_min_span():
    c = cfg(min_span=timedelta(days=3))
    assert evaluate(claim(), tl(S=1.4, keys=2, span_days=0), "t", c).decision == "provisional"
    assert evaluate(claim(), tl(S=1.4, keys=2, span_days=4), "t", c).decision == "established"


def test_established_disputed_by_contradiction_ratio():
    t = tl(S=1.4, C=1.0, keys=2, contra_keys=1)
    assert evaluate(claim("established"), t, "t", CFG).decision == "disputed"
    assert evaluate(claim("disputed"), t, "t", CFG).decision == "disputed"
    assert evaluate(claim("candidate"), t, "t", CFG).decision == "provisional"
    one = tl(S=0.7, C=0.7, keys=1, contra_keys=1)
    assert evaluate(claim("candidate"), one, "t", CFG).decision == "candidate"


def test_disputed_recovers():
    r = evaluate(claim("disputed"), tl(S=1.4, keys=2), "t", CFG)
    assert r.decision == "established"


def test_verified_contradiction_deprecates_only_when_c_ge_s():
    t = tl(S=0.7, C=0.7, keys=1, contra_keys=1, vc=True)
    assert evaluate(claim("established"), t, "t", CFG).decision == "deprecated"
    t = tl(S=1.4, C=0.7, keys=2, contra_keys=1, vc=True)
    assert evaluate(claim("established"), t, "t", CFG).decision != "deprecated"


def test_high_risk_needs_human():
    c = replace(CFG, high_risk_terms=("delete",))
    text = "Delete the staging DB first"
    r = evaluate(claim(), tl(S=1.4, keys=2), text, c)
    assert r.decision == "provisional"
    assert r.checks["high_risk"] is False
    r = evaluate(claim(), tl(S=1.4, keys=2), "restart it", c)
    assert r.checks["high_risk"] is True
    h = tl(S=3.7, keys=2, verified=1, by={"human": 1, "observation": 1})
    assert evaluate(claim(), h, text, c).decision == "established"


def test_unknown_scope_raises():
    c = claim()
    c.scope = "org"
    with pytest.raises(ValueError, match="org"):
        evaluate(c, tl(), "t", CFG)


def test_report_json_is_plain():
    r = evaluate(claim(), tl(S=1.4, keys=2), "t", CFG)
    assert isinstance(r, GateReport)
    j = r.to_json()
    assert j["decision"] == "established"
    assert j["checks"]["sources"] is True
    assert isinstance(j["S"], float) and isinstance(j["confidence"], float)
    import json
    json.dumps(j)
