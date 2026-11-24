"""Hand-off: from a query to at most k items, one per claim, each with all
its supporting evidence as sources (spec section 11). Reads only."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from cortex_mem.access import event_visible
from cortex_mem.contract import Caller, EngineContext, Event, Handoff, Scope
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.evidence import Tally, tally
from cortex_mem.engines.ledger.model import Claim, Evidence
from cortex_mem.engines.ledger.similar import similar_of
from cortex_mem.engines.ledger.store import LedgerStore

_RRF = 60
_CHANNELS = ("human", "test", "doc", "review", "observation")


def provenance_line(claim: Claim, t: Tally) -> str:
    """`[established · human 1, test 1 · 2 sources · since 2026-10-03]`."""
    channels = ", ".join(
        f"{ch} {t.by_channel[ch]}" for ch in _CHANNELS if t.by_channel.get(ch)
    )
    since = t.first_support.date().isoformat() if t.first_support else "unknown"
    parts = [claim.status]
    if channels:
        parts.append(channels)
    parts.append(f"{t.support_keys} sources")
    parts.append(f"since {since}")
    line = " · ".join(parts)
    if claim.status == "disputed":
        line += f" · {t.contra_keys} against"
    return f"[{line}]"


def _candidates(
    query: str, caller: Caller, scope: Scope, ctx: EngineContext,
    store: LedgerStore, cfg: LedgerConfig,
) -> dict[str, float]:
    """Claim id -> relevance, summing 1 / (60 + rank) over the lists in which
    the claim first appears at that rank (rank starts at 1)."""
    lists: list[list[str]] = [
        [item.id for item, _ in ctx.knowledge.search(query, caller, scope, cfg.pool)],
        _claims_of(ctx.events.search(query, caller, scope, cfg.pool, ["record"]), store),
    ]
    port = similar_of(ctx)
    if port is not None:
        lists.append([item.id for item, _ in port.knowledge(query, caller, scope, cfg.pool)])
        lists.append(_claims_of(port.search(query, caller, scope, cfg.pool, ["record"]), store))
    relevance: dict[str, float] = {}
    for ids in lists:
        seen: set[str] = set()
        rank = 0
        for claim_id in ids:
            if claim_id in seen:
                continue
            seen.add(claim_id)
            rank += 1
            relevance[claim_id] = relevance.get(claim_id, 0.0) + 1.0 / (_RRF + rank)
    return relevance


def _claims_of(found: Sequence[tuple[Event, float]], store: LedgerStore) -> list[str]:
    """Claim ids of the observed records among search hits, in hit order."""
    ids: list[str] = []
    for event, _ in found:
        memory = store.memory(event.id)
        if memory is not None:
            ids.append(memory.claim_id)
    return ids


def handoff(
    query: str, caller: Caller, scope: Scope, ctx: EngineContext,
    store: LedgerStore, cfg: LedgerConfig, k: int,
) -> list[Handoff]:
    """At most k items, one per claim, best first.

    Candidates are claims fused by RRF over knowledge and record search (plus
    the similarity port's two lists when embeddings are on). Deprecated and
    archived claims, claims with a condition `caller.context` contradicts,
    claims whose canonical record the caller cannot see, and claims with no
    counted support are left out. Score = relevance x status weight (the
    caller's own candidate uses `own_candidate_weight`) x recency; ties by
    claim id. `status` is "knowledge" only for an established claim the
    knowledge base holds."""
    # Score every candidate from its claim and canonical record alone, then read
    # evidence best first and only until k items are found: evidence decides
    # whether a claim is left out (no counted support) but not its score.
    claims: list[tuple[Claim, float]] = []
    for claim_id, relevance in _candidates(query, caller, scope, ctx, store, cfg).items():
        claim = store.claim(claim_id)
        if claim is None or claim.status in ("deprecated", "archived"):
            continue
        if any(caller.context.get(key, value) != value for key, value in claim.conditions.items()):
            continue
        claims.append((claim, relevance))
    canonicals = {e.id: e for e in ctx.events.get([c.canonical for c, _ in claims])}

    ranked: list[tuple[float, Claim, Event]] = []
    for claim, relevance in claims:
        canonical = canonicals.get(claim.canonical)
        if canonical is None or not event_visible(canonical, caller, scope):
            continue
        own = claim.status == "candidate" and canonical.user == caller.user
        weight = cfg.own_candidate_weight if own else cfg.status_weights[claim.status]
        ranked.append((relevance * weight * _recency(claim, caller, cfg), claim, canonical))
    ranked.sort(key=lambda r: (-r[0], r[1].id))

    chosen: list[tuple[Claim, Event, list[Evidence], Tally]] = []
    for _, claim, canonical in ranked:
        if len(chosen) == k:
            break
        evidence = store.evidence(claim.id)
        support = [e for e in evidence if e.stance == "support" and e.derived is None]
        if support:
            chosen.append((claim, canonical, support, tally(evidence, cfg)))

    established = [c.id for c, _, _, _ in chosen if c.status == "established"]
    held = {item.id for item in ctx.knowledge.get(established)} if established else set()
    return [
        Handoff(
            id=claim.id,
            content=f"{canonical.text}\n\n{provenance_line(claim, full)}",
            sources=tuple(e.event_id for e in support),
            status="knowledge" if claim.id in held else "memory",
            conditions=dict(claim.conditions),
            confidence=full.confidence,
        )
        for claim, canonical, support, full in chosen
    ]


def _recency(claim: Claim, caller: Caller, cfg: LedgerConfig) -> float:
    """Recency weight: 0.5 ** (age / half_life), where age is the time since
    the last counted support (0 if that is after `caller.at`). It falls from
    1.0 to 0.5 over one half-life and stays at 0.5 after that, so recency at
    most halves a score. 1.0 when `half_life_days` is 0."""
    if not cfg.half_life:
        return 1.0
    age = max(caller.at - claim.last_support_at, timedelta(0))
    return max(0.5, 0.5 ** (age / cfg.half_life))
