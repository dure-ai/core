"""Which existing claim a new record joins (spec section 7).

Matchers only read: registering groups and signatures is the engine's job.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from cortex_mem.access import owner_for
from cortex_mem.contract import Caller, EngineContext, Event
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.model import Claim, Memory
from cortex_mem.engines.ledger.similar import similar_of
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.engines.ledger.text import jaccard, signature

_DOWN = ("deprecated", "archived")


class Match(NamedTuple):
    claim_id: str
    similarity: float
    copied_from: str | None  # event id of another author's record copied verbatim


class ClaimMatcher(Protocol):
    def match(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> Match | None: ...


def _author(event: Event) -> Caller:
    return Caller(user=event.user, team=event.team, session=event.session, at=event.at)


def _eligible(claim: Claim | None, event: Event) -> bool:
    """Same (scope, owner, conditions) and not taken down."""
    return (
        claim is not None
        and claim.status not in _DOWN
        and claim.scope == event.scope
        and claim.owner == owner_for(event)
        and claim.conditions == dict(event.conditions)
    )


def _copied_from(
    event: Event, claim: Claim, ctx: EngineContext, store: LedgerStore, cfg: LedgerConfig
) -> str | None:
    ids = {claim.id.removeprefix("c-"), claim.canonical}
    ids.update(ev.event_id for ev in store.evidence(claim.id))
    others = [
        i for i in sorted(ids)
        if (m := store.memory(i)) is not None
        and m.claim_id == claim.id
        and m.user != event.user
    ]
    best: tuple[float, str] | None = None
    for other in ctx.events.get(others):
        score = jaccard(event.text, other.text)
        if score >= cfg.copy_threshold and (best is None or score > best[0]):
            best = (score, other.id)  # ids ascend, so the first of equal scores wins
    return None if best is None else best[1]


class GroupMatcher:
    def __init__(self, cfg: LedgerConfig) -> None:
        self._cfg = cfg

    def match(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> Match | None:
        similar = similar_of(ctx)
        if similar is None:
            return None
        group = similar.group(event.id)
        if group is None:
            return None
        candidates: dict[str, Claim] = {}
        for cid in store.claims_in_group(group.id):
            claim = store.claim(cid)
            if claim is not None and _eligible(claim, event):
                candidates[cid] = claim
        if not candidates:
            return None
        scores: dict[str, float] = {}
        neighbors = similar.neighbors(
            event.id, _author(event), event.scope, k=self._cfg.match_pool
        )
        for other, score in neighbors:
            memory = store.memory(other.id)
            if memory is not None and memory.claim_id in candidates:
                cid = memory.claim_id
                scores[cid] = max(scores.get(cid, score), score)
        ranked = sorted(
            ((scores.get(cid, 0.0), cid) for cid in candidates),
            key=lambda p: (-p[0], p[1]),
        )
        for score, cid in ranked:
            if score >= self._cfg.similar_same_threshold:
                copied = _copied_from(event, candidates[cid], ctx, store, self._cfg)
                return Match(cid, score, copied)
        return None


class LexicalMatcher:
    def __init__(self, cfg: LedgerConfig) -> None:
        self._cfg = cfg

    def match(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> Match | None:
        owner = owner_for(event)
        cid = store.claim_by_signature(event.scope, owner, signature(event.text))
        if cid is not None:
            claim = store.claim(cid)
            if claim is not None and _eligible(claim, event):
                return Match(cid, 1.0, _copied_from(event, claim, ctx, store, self._cfg))
        best: tuple[float, str, str] | None = None  # (score, claim id, event id)
        for other, memory in self._observed_hits(event, ctx, store):
            claim = store.claim(memory.claim_id)
            if not _eligible(claim, event):
                continue
            score = jaccard(event.text, other.text)
            if score < self._cfg.same_threshold:
                continue
            key = (score, memory.claim_id, other.id)
            if best is None or (-key[0], key[1], key[2]) < (-best[0], best[1], best[2]):
                best = key
        if best is None:
            return None
        claim = store.claim(best[1])
        assert claim is not None
        return Match(best[1], best[0], _copied_from(event, claim, ctx, store, self._cfg))

    def _observed_hits(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[tuple[Event, Memory]]:
        """The first `match_pool` search hits that the engine has observed
        (they have a memory), in search order, never the event itself.

        `ctx.events` is the whole log, not bounded to this event like
        `ctx.similar` (spec 13): records later in the log take pool slots
        during a batched sync or a rebuild. Skipping them and widening the
        search (doubling k) until the pool is full or the search runs out
        keeps a replay matching like live delivery."""
        pool = self._cfg.match_pool
        k = pool
        while True:
            hits = ctx.events.search(
                event.text, _author(event), event.scope, k=k, kinds=("record",)
            )
            observed: list[tuple[Event, Memory]] = []
            for other, _ in hits:
                if other.id == event.id:
                    continue
                memory = store.memory(other.id)
                if memory is not None:
                    observed.append((other, memory))
                    if len(observed) == pool:
                        return observed
            if len(hits) < k:
                return observed
            k *= 2


def group_of(event: Event, ctx: EngineContext) -> str | None:
    """The core group id of the record, or None without a port or a group."""
    similar = similar_of(ctx)
    if similar is None:
        return None
    group = similar.group(event.id)
    return None if group is None else group.id


def match_claim(
    event: Event, ctx: EngineContext, store: LedgerStore, cfg: LedgerConfig
) -> tuple[Match | None, str | None]:
    """(the claim found, the record's core group id)."""
    matcher: ClaimMatcher = (
        GroupMatcher(cfg) if similar_of(ctx) is not None else LexicalMatcher(cfg)
    )
    return matcher.match(event, ctx, store), group_of(event, ctx)
