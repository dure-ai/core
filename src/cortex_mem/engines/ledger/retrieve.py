"""Hand-off: from a query to at most k items, one per claim, each with all
its supporting evidence as sources (spec section 11). Reads only."""

from __future__ import annotations

from collections.abc import Sequence

from cortex_mem.access import event_visible
from cortex_mem.contract import Caller, EngineContext, Handoff, Scope
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


def _claims_of(found: Sequence[tuple[object, float]], store: LedgerStore) -> list[str]:
    ids: list[str] = []
    for event, _ in found:
        memory = store.memory(event.id)  # type: ignore[attr-defined]
        if memory is not None:
            ids.append(memory.claim_id)
    return ids


def handoff(
    query: str, caller: Caller, scope: Scope, ctx: EngineContext,
    store: LedgerStore, cfg: LedgerConfig, k: int,
) -> list[Handoff]:
    ranked: list[tuple[float, str, Handoff]] = []
    for claim_id, relevance in _candidates(query, caller, scope, ctx, store, cfg).items():
        claim = store.claim(claim_id)
        if claim is None or claim.status in ("deprecated", "archived"):
            continue
        if any(caller.context.get(key, value) != value for key, value in claim.conditions.items()):
            continue
        found = ctx.events.get([claim.canonical])
        if not found or not event_visible(found[0], caller, scope):
            continue
        canonical = found[0]
        evidence = store.evidence(claim.id)
        support: list[Evidence] = [e for e in evidence if e.stance == "support" and e.derived is None]
        if not support:
            continue
        line = provenance_line(claim, _line_tally(claim, evidence, support, cfg))
        full = tally(evidence, cfg)
        own = claim.status == "candidate" and canonical.user == caller.user
        weight = cfg.status_weights[claim.status]
        if own:
            weight = cfg.own_candidate_weight
        score = relevance * weight * _recency(claim, caller, cfg)
        knowledge = (
            claim.status == "established" and bool(ctx.knowledge.get([claim.id]))
        )
        ranked.append(
            (
                score,
                claim.id,
                Handoff(
                    id=claim.id,
                    content=f"{canonical.text}\n\n{line}",
                    sources=tuple(e.event_id for e in support),
                    status="knowledge" if knowledge else "memory",
                    conditions=dict(claim.conditions),
                    confidence=full.confidence,
                ),
            )
        )
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return [h for _, _, h in ranked[:k]]


def _line_tally(
    claim: Claim, evidence: Sequence[Evidence], support: list[Evidence], cfg: LedgerConfig
) -> Tally:
    """Support counted from the sources only; contradictions from all evidence."""
    counted = tally(support, cfg)
    contra = tally(evidence, cfg)
    return Tally(
        S=counted.S, C=contra.C, support_keys=counted.support_keys,
        verified_keys=counted.verified_keys, contra_keys=contra.contra_keys,
        verified_contra=contra.verified_contra, first_support=counted.first_support,
        last_support=counted.last_support, by_channel=counted.by_channel,
        confidence=contra.confidence,
    )


def _recency(claim: Claim, caller: Caller, cfg: LedgerConfig) -> float:
    if not cfg.half_life:
        return 1.0
    age = max(caller.at - claim.last_support_at, caller.at - caller.at)
    return max(0.5, 0.5 ** (age / cfg.half_life))
