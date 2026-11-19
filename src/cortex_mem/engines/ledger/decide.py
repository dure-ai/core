"""Turn claim status changes into Promote/Demote decisions (spec section 9).

Pure module apart from updating the claim it is given: no clock, no
randomness, no state access.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from cortex_mem.contract import Decision, Demote, Event, Knowledge, Promote
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.model import Claim, ClaimStatus, Evidence

TITLE_LENGTH = 80


def _sources(evidence: Sequence[Evidence]) -> tuple[str, ...]:
    """Event ids of the counted support evidence, oldest first."""
    rows = [e for e in evidence if e.stance == "support" and e.derived is None]
    rows.sort(key=lambda e: (e.at, e.event_id))
    return tuple(dict.fromkeys(e.event_id for e in rows))


def knowledge_item(claim: Claim, evidence: Sequence[Evidence], text: str) -> Knowledge:
    """The knowledge base item of a claim; `text` is its canonical record's text."""
    lines = text.strip().splitlines()
    title = lines[0].strip() if lines else claim.id
    return Knowledge(
        id=claim.id,
        kind="knowledge",
        title=title[:TITLE_LENGTH],
        body=text,
        scope=claim.scope,
        owner=claim.owner,
        sources=_sources(evidence),
        conditions=dict(claim.conditions),
    )


def canonical_of(
    evidence: Sequence[Evidence],
    cfg: LedgerConfig,
    *,
    records: Collection[str] | None = None,
) -> str:
    """Event id of the claim's canonical record (the source of its text).

    The counted support evidence of a record with the largest weight x trust,
    then the latest, then the largest event id. `records` are the event ids
    of the claim's records (its memories): evidence from any other event (a
    judgement) is passed over. None treats every row as a record's.

    With no counted support record, the claim's first record: the earliest
    record row of any kind (every record writes one)."""

    def is_record(e: Evidence) -> bool:
        return records is None or e.event_id in records

    def strength(e: Evidence) -> float:
        trust = cfg.trust_observation if e.channel == "observation" else 1.0
        return cfg.weights[e.channel] * trust

    counted = [
        e for e in evidence if e.stance == "support" and e.derived is None and is_record(e)
    ]
    if counted:
        return max(counted, key=lambda e: (strength(e), e.at, e.event_id)).event_id
    rows = [e for e in evidence if is_record(e)]
    if not rows:
        raise ValueError("a claim with no record evidence has no canonical record")
    return min(rows, key=lambda e: (e.at, e.event_id)).event_id


def decisions(
    before: ClaimStatus,
    canonical_before: str,
    claim: Claim,
    evidence: Sequence[Evidence],
    text: str,
    trigger: Event,
) -> list[Decision]:
    """Decisions for a claim whose status (and canonical record) were just
    re-evaluated from `before` / `canonical_before`. Updates `claim.promoted`
    and, when it becomes deprecated, `claim.invalid_since`."""
    if claim.status == "established":
        if before != "established" or claim.canonical != canonical_before:
            claim.promoted = True
            return [Promote(knowledge_item(claim, evidence, text))]
        return []
    if claim.status == "deprecated" and before != "deprecated":
        claim.invalid_since = trigger.at
        if not claim.promoted:
            return []
        claim.promoted = False
        report = claim.report
        reason = (
            "verified contradiction outweighs support "
            f"(C={float(report.get('C', 0.0)):.2f} >= S={float(report.get('S', 0.0)):.2f})"
        )
        return [
            Demote(claim.id, reason, (trigger.id,), why="invalid", since=trigger.at)
        ]
    return []
