"""Promotion gate and claim status decisions (spec section 9).

Pure module: no clock, no randomness, no state access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.evidence import Tally
from cortex_mem.engines.ledger.model import Claim, ClaimStatus


@dataclass(frozen=True)
class GateReport:
    checks: Mapping[str, bool]  # sources, verified, span, contradiction, high_risk
    S: float
    C: float
    confidence: float
    decision: ClaimStatus

    def to_json(self) -> dict[str, Any]:
        return {
            "checks": {k: bool(v) for k, v in self.checks.items()},
            "S": float(self.S),
            "C": float(self.C),
            "confidence": float(self.confidence),
            "decision": str(self.decision),
        }


def evaluate(claim: Claim, t: Tally, text: str, cfg: LedgerConfig) -> GateReport:
    """Run the gate checks for a claim and decide its next status."""
    rule = cfg.gates.get(claim.scope)
    if rule is None:
        raise ValueError(f"no gate rule for scope: {claim.scope!r}")

    sources = t.support_keys >= rule.min_sources or (
        rule.alt_verified > 0 and t.verified_keys >= rule.alt_verified
    )
    verified = (not rule.require_verified) or t.verified_keys >= 1
    span = (
        t.first_support is not None
        and t.last_support is not None
        and t.last_support - t.first_support >= rule.min_span
    )
    total = t.S + t.C
    contradiction = total == 0 or t.C / total <= rule.max_contra_ratio
    lowered = text.lower()
    high_risk = (
        not any(term in lowered for term in cfg.high_risk_terms)
        or t.by_channel.get("human", 0) > 0
    )
    checks = {
        "sources": sources,
        "verified": verified,
        "span": span,
        "contradiction": contradiction,
        "high_risk": high_risk,
    }

    decision: ClaimStatus
    if t.verified_contra and t.C >= t.S:
        decision = "deprecated"
    elif all(checks.values()):
        decision = "established"
    elif not contradiction and claim.status in ("established", "disputed"):
        decision = "disputed"
    elif sources:
        decision = "provisional"
    else:
        decision = "candidate"
    if decision in ("provisional", "candidate") and claim.status in (
        "established",
        "disputed",
    ):
        # A promoted claim moves only by the table's transitions (spec 9):
        # failing sources, verified, span or high_risk never demotes it.
        decision = claim.status
    return GateReport(checks, t.S, t.C, t.confidence, decision)
