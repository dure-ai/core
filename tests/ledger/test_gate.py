from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.evidence import Tally, tally
from cortex_mem.engines.ledger.gate import GateReport, evaluate
from cortex_mem.engines.ledger.model import Claim, Evidence

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
    r = evaluate(claim(), tl(), "text", CFG, human_fresh=False)
    assert r.decision == "candidate"
    assert r.checks["sources"] is False


def test_two_people_established():
    r = evaluate(claim(), tl(S=1.4, keys=2), "text", CFG, human_fresh=False)
    assert r.decision == "established"
    assert all(r.checks.values())


def test_one_person_with_test_established_by_alt_verified():
    r = evaluate(claim(), tl(S=2.0, keys=1, verified=1, by={"test": 1}), "text", CFG, human_fresh=False)
    assert r.decision == "established"


def test_require_verified_leaves_provisional():
    r = evaluate(claim(), tl(S=1.4, keys=2), "text", cfg(require_verified=True), human_fresh=False)
    assert r.decision == "provisional"
    assert r.checks["verified"] is False


def test_min_span():
    c = cfg(min_span=timedelta(days=3))
    assert evaluate(claim(), tl(S=1.4, keys=2, span_days=0), "t", c, human_fresh=False).decision == "provisional"
    assert evaluate(claim(), tl(S=1.4, keys=2, span_days=4), "t", c, human_fresh=False).decision == "established"


def test_established_disputed_by_contradiction_ratio():
    t = tl(S=1.4, C=1.0, keys=2, contra_keys=1)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision == "disputed"
    assert evaluate(claim("disputed"), t, "t", CFG, human_fresh=False).decision == "disputed"
    assert evaluate(claim("candidate"), t, "t", CFG, human_fresh=False).decision == "provisional"
    one = tl(S=0.7, C=0.7, keys=1, contra_keys=1)
    assert evaluate(claim("candidate"), one, "t", CFG, human_fresh=False).decision == "candidate"


def test_disputed_recovers():
    r = evaluate(claim("disputed"), tl(S=1.4, keys=2), "t", CFG, human_fresh=False)
    assert r.decision == "established"


def test_verified_contradiction_deprecates_only_when_c_ge_s():
    t = tl(S=0.7, C=0.7, keys=1, contra_keys=1, vc=True)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision == "deprecated"
    t = tl(S=1.4, C=0.7, keys=2, contra_keys=1, vc=True)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision != "deprecated"


def test_high_risk_needs_human():
    c = replace(CFG, high_risk_terms=("delete",))
    text = "Delete the staging DB first"
    r = evaluate(claim(), tl(S=1.4, keys=2), text, c, human_fresh=False)
    assert r.decision == "provisional"
    assert r.checks["high_risk"] is False
    r = evaluate(claim(), tl(S=1.4, keys=2), "restart it", c, human_fresh=False)
    assert r.checks["high_risk"] is True
    h = tl(S=3.7, keys=2, verified=1, by={"human": 1, "observation": 1})
    assert evaluate(claim(), h, text, c, human_fresh=True).decision == "established"


def test_unknown_scope_raises():
    c = claim()
    c.scope = "org"
    with pytest.raises(ValueError, match="org"):
        evaluate(c, tl(), "t", CFG, human_fresh=False)


def test_report_json_is_plain():
    r = evaluate(claim(), tl(S=1.4, keys=2), "t", CFG, human_fresh=False)
    assert isinstance(r, GateReport)
    j = r.to_json()
    assert j["decision"] == "established"
    assert j["checks"]["sources"] is True
    assert isinstance(j["S"], float) and isinstance(j["confidence"], float)
    import json
    json.dumps(j)


def test_established_disputed_even_when_span_fails():
    c = cfg(min_span=timedelta(days=3))
    t = tl(S=1.4, C=1.0, keys=2, contra_keys=1, span_days=0)
    assert evaluate(claim("established"), t, "t", c, human_fresh=False).decision == "disputed"


def test_established_disputed_even_when_sources_fail():
    t = tl(S=0.7, C=0.7, keys=1, contra_keys=1)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision == "disputed"


def test_disputed_stays_disputed_when_verified_fails():
    t = tl(S=1.4, C=1.0, keys=2, contra_keys=1)
    r = evaluate(claim("disputed"), t, "t", cfg(require_verified=True), human_fresh=False)
    assert r.decision == "disputed"


@pytest.mark.parametrize("status", ["established", "disputed"])
def test_promoted_claims_do_not_fall_back_when_checks_fail(status):
    # no sources, no verified channel, no span, a high-risk text: the checks
    # that gate a promotion fail, but a promoted claim moves only by the table
    t = tl(S=0.7, C=0.0, keys=1)
    c = cfg(min_span=timedelta(days=3), require_verified=True)
    risky = replace(c, high_risk_terms=("password",))
    r = evaluate(claim(status), t, "the password is x", risky, human_fresh=False)
    assert r.decision == status
    assert r.checks["sources"] is False and r.checks["high_risk"] is False


def test_established_without_sources_stays_established_but_unpromoted_does_not():
    t = tl(S=0.7, keys=1)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision == "established"
    assert evaluate(claim("candidate"), t, "t", CFG, human_fresh=False).decision == "candidate"
    t2 = tl(S=1.4, keys=2)
    low = cfg(min_span=timedelta(days=3))
    assert evaluate(claim("provisional"), t2, "t", low, human_fresh=False).decision == "provisional"
    assert evaluate(claim("established"), t2, "t", low, human_fresh=False).decision == "established"


def test_promoted_claims_still_deprecate_on_strong_contradiction():
    t = tl(S=1.4, C=2.0, keys=2, contra_keys=1, vc=True)
    assert evaluate(claim("established"), t, "t", CFG, human_fresh=False).decision == "deprecated"
    assert evaluate(claim("disputed"), t, "t", CFG, human_fresh=False).decision == "deprecated"


def test_high_risk_needs_fresh_human_support_not_just_any():
    t = tl(S=3.0, keys=2, verified=1, by={"human": 1, "observation": 1})
    risky = replace(CFG, high_risk_terms=("password",))
    stale = evaluate(claim(), t, "the password", risky, human_fresh=False)
    assert stale.checks["high_risk"] is False and stale.decision != "established"
    fresh = evaluate(claim(), t, "the password", risky, human_fresh=True)
    assert fresh.checks["high_risk"] is True and fresh.decision == "established"
    benign = evaluate(claim(), t, "all fine", risky, human_fresh=False)
    assert benign.checks["high_risk"] is True


def test_human_fresh_must_be_given():
    with pytest.raises(TypeError, match="human_fresh"):
        evaluate(claim(), tl(), "t", CFG)  # type: ignore[call-arg]


def _evidence(stance: str, counts: dict[str, int], tag: str) -> list[Evidence]:
    # Keys sort observation first, then review, test, human: the order tally adds them in.
    order = ("observation", "review", "test", "human")
    rows = []
    for rank, channel in enumerate(order):
        for i in range(counts.get(channel, 0)):
            key = f"u:{tag}{rank}{channel}{i}"
            rows.append(Evidence(claim_id="c1", event_id=f"{key}-{stance}", stance=stance,
                                 channel=channel, key=key, user=key, at=T0, derived=None))
    return rows


def test_a_contradiction_ratio_exactly_at_the_limit_passes():
    # S = 4*0.7 + 1.5 + 2.0 + 3*3.0 = 15.3, C = 0.7 + 2.0 = 2.7; C/(S+C) = 0.15 exactly,
    # which a plain float division puts just above the team limit of 0.15.
    rows = _evidence("support", {"observation": 4, "review": 1, "test": 1, "human": 3}, "s")
    rows += _evidence("contradict", {"observation": 1, "test": 1}, "c")
    r = evaluate(claim(), tally(rows, CFG), "text", CFG, human_fresh=False)
    assert r.checks["contradiction"] is True
    assert r.decision == "established"


def _in_key_order(stance: str, channels: str, tag: str) -> list[Evidence]:
    names = {"o": "observation", "r": "review", "t": "test", "h": "human"}
    rows = []
    for i, c in enumerate(channels):
        key = f"u:{tag}{i:02d}"
        rows.append(Evidence(claim_id="c1", event_id=f"{key}-{stance}", stance=stance,
                             channel=names[c], key=key, user=key, at=T0, derived=None))
    return rows


def test_verified_contradiction_equal_to_support_deprecates():
    # S = 0.7 + 1.5 + 0.7 = 2.9 and C = 0.7 + 0.7 + 1.5 = 2.9 exactly. Added in key order
    # the float S is 2.9000000000000004, so a plain C >= S misses the tie.
    rows = _in_key_order("support", "oro", "s") + _in_key_order("contradict", "oor", "c")
    t = tally(rows, CFG)
    assert t.C < t.S  # the float sums really do miss the tie
    r = evaluate(claim(status="established"), t, "text", CFG, human_fresh=False)
    assert r.decision == "deprecated"


def test_high_risk_terms_match_words_not_parts_of_words():
    c = replace(CFG, high_risk_terms=("key", "rm"))
    two = tl(S=1.4, keys=2)
    assert evaluate(claim(), two, "The monkey patch in form.py", c,
                    human_fresh=False).checks["high_risk"] is True
    assert evaluate(claim(), two, "Rotate the API key", c,
                    human_fresh=False).checks["high_risk"] is False
    assert evaluate(claim(), two, "Export DB_PASSWORD first", replace(c, high_risk_terms=("password",)),
                    human_fresh=False).checks["high_risk"] is False
