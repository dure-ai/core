"""LedgerEngine: records become claims, claims collect evidence, and claims
that pass the gate become knowledge (spec sections 4, 6 and 9).

The engine keeps nothing on the instance but its configuration: everything
durable is in ctx.state, so observing one event twice (contract 0.4.0) has
no effect beyond the state and the decisions returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from cortex_mem.access import owner_for
from cortex_mem.contract import (
    Caller,
    Decision,
    EngineContext,
    Event,
    Handoff,
    Scope,
)
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.decide import canonical_of, decisions
from cortex_mem.engines.ledger.evidence import channel_of, independence_key, tally
from cortex_mem.engines.ledger.gate import evaluate
from cortex_mem.engines.ledger.matcher import match_claim
from cortex_mem.engines.ledger.model import Claim, Derived, Evidence, Memory
from cortex_mem.engines.ledger.similar import similar_of
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.engines.ledger.text import signature


class LedgerEngine:
    state_version = "1"

    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        self._cfg = LedgerConfig.from_mapping(config)

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        store = LedgerStore(ctx.state)
        if event.kind == "record":
            return self._record(event, ctx, store)
        return []

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        return []

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        return []

    def _record(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Spec section 6, `record`: find or open the claim, write the memory
        and its support evidence, then re-evaluate the claim."""
        if event.scope == "org":
            store.skip("org_scope")  # v0.1 covers personal and team only
            return []
        if store.memory(event.id) is not None:
            return []  # delivered again: this record is already in the ledger

        match, group_id = match_claim(event, ctx, store, self._cfg)
        if match is None:
            claim = Claim(
                id=f"c-{event.id}",
                scope=event.scope,
                owner=owner_for(event),
                conditions=dict(event.conditions),
                canonical=event.id,
                status="candidate",
                first_at=event.at,
                last_support_at=event.at,
                invalid_since=None,
                promoted=False,
                use_success=0,
                use_failure=0,
            )
            store.put_claim(claim)
        else:
            found = store.claim(match.claim_id)
            assert found is not None, f"matcher returned unknown claim {match.claim_id!r}"
            claim = found
        if group_id is not None:
            store.add_to_group(group_id, claim.id)
        if similar_of(ctx) is None:
            store.index_signature(
                claim.scope, claim.owner, signature(event.text), claim.id
            )

        derived: Derived | None = None
        if match is not None:
            if self._retrieved(event, claim.id, store):
                derived = "retrieved"
            elif match.copied_from is not None:
                derived = "copy"

        store.put_memory(
            Memory(
                id=event.id,
                claim_id=claim.id,
                user=event.user,
                team=event.team,
                session=event.session,
                at=event.at,
                group=group_id,
            )
        )
        store.put_evidence(
            Evidence(
                claim_id=claim.id,
                event_id=event.id,
                stance="support",
                channel=channel_of(event),
                key=independence_key(claim.scope, event),
                user=event.user,
                at=event.at,
                derived=derived,
            )
        )
        return self._reevaluate(claim, event, ctx, store)

    @staticmethod
    def _retrieved(event: Event, claim_id: str, store: LedgerStore) -> bool:
        """The record repeats what it was handed: it cites the claim, or its
        session was handed the claim before the record was made."""
        if claim_id in event.refs:
            return True
        seen = store.seen(event.session, claim_id)
        return seen is not None and seen < event.at

    def _reevaluate(
        self, claim: Claim, trigger: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Tally the claim's evidence, run the gate, update the claim and
        return the decisions its change of status or canonical record calls
        for. Writes the claim."""
        before, canonical_before = claim.status, claim.canonical
        evidence = store.evidence(claim.id)
        records = {
            e.event_id
            for e in evidence
            if (m := store.memory(e.event_id)) is not None and m.claim_id == claim.id
        }
        canonical = canonical_of(
            evidence, self._cfg, records=records, current=canonical_before
        )
        found = ctx.events.get([canonical])
        text = found[0].text if found else ""

        counted = tally(evidence, self._cfg)
        report = evaluate(claim, counted, text, self._cfg)
        claim.status = report.decision
        claim.report = report.to_json()
        claim.canonical = canonical
        if counted.first_support is not None:
            claim.first_at = counted.first_support
        if counted.last_support is not None:
            claim.last_support_at = counted.last_support

        out = decisions(before, canonical_before, claim, evidence, text, trigger)
        store.put_claim(claim)
        return out
