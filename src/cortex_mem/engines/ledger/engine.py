"""LedgerEngine: records become claims, claims collect evidence, and claims
that pass the gate become knowledge (spec sections 4, 6 and 9).

The engine keeps nothing on the instance but its configuration: everything
durable is in ctx.state, so observing one event twice (contract 0.4.0) has
no effect beyond the state and the decisions returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from cortex_mem.access import owner_for
from cortex_mem.contract import (
    Caller,
    Decision,
    Demote,
    EngineContext,
    Event,
    Handoff,
    Scope,
)
from cortex_mem.engines.ledger import retrieve
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.decide import canonical_of, decisions
from cortex_mem.engines.ledger.evidence import (
    VERIFIED,
    channel_of,
    in_scope,
    independence_key,
    tally,
)
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
        if event.kind in ("outcome", "human_confirm"):
            return self._judge(event, ctx, store)
        if event.kind == "recall":
            for claim_id in self._resolve(event.refs, store, count_unknown=False):
                store.mark_seen(event.session, claim_id, event.at)
        return []

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        """Spec section 10: when `review_after` is set, archive each established
        claim whose last support is older than it, and demote its knowledge.
        Claims are decided in id order; the result is idempotent because an
        archived claim is no longer established."""
        if not self._cfg.review_after:
            return []
        store = LedgerStore(ctx.state)
        review_days = self._cfg.review_after / timedelta(days=1)
        out: list[Decision] = []
        for claim in store.claims():
            if claim.status != "established":
                continue
            if now - claim.last_support_at <= self._cfg.review_after:
                continue
            claim.status = "archived"
            if claim.promoted:
                claim.promoted = False
                out.append(
                    Demote(
                        claim.id,
                        f"no support for over {review_days:g} days",
                        (claim.canonical,),
                        why="stale",
                        since=now,
                    )
                )
            store.put_claim(claim)
        return out

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        store = LedgerStore(ctx.state)
        return retrieve.handoff(query, caller, scope, ctx, store, self._cfg, k)

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
    def _resolve(
        refs: Sequence[str], store: LedgerStore, *, count_unknown: bool = True
    ) -> list[str]:
        """Claim ids the refs name: a claim id as it is, an event id through
        its memory. Unique, in order; unknown refs are counted."""
        found: dict[str, None] = {}
        for ref in refs:
            if store.claim(ref) is not None:
                found[ref] = None
            elif (memory := store.memory(ref)) is not None:
                found[memory.claim_id] = None
            elif count_unknown:
                store.skip("unknown_ref")
        return list(found)

    def _judge(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Spec section 6, `outcome` and `human_confirm`: add support or
        contradiction to the claims the refs name, when in scope."""
        if event.scope == "org":
            # v0.1 covers personal and team only: an org judgment would let
            # other teams vote and hide the claim from team-scope handoffs
            store.skip("org_scope")
            return []
        if event.outcome not in ("success", "failure"):
            store.skip("unknown_outcome")
            return []
        out: list[Decision] = []
        for claim_id in self._resolve(event.refs, store):
            claim = store.claim(claim_id)
            assert claim is not None
            if not in_scope(event, claim):
                store.skip("out_of_scope_evidence")
                continue
            if store.judged(claim_id, event.id):
                continue  # delivered again: this judgment is already counted
            store.mark_judged(claim_id, event.id)
            out += self._judge_claim(claim, event, ctx, store)
        return out

    def _judge_claim(
        self, claim: Claim, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        success = event.outcome == "success"
        own_report = event.kind == "outcome" and event.signal in (None, "self")
        if own_report and success:
            # usefulness, not truth: no vote
            if store.seen(event.session, claim.id) is not None:
                claim.use_success += 1
                store.put_claim(claim)
            return []
        if own_report:
            claim.use_failure += 1
        channel = channel_of(event)
        store.put_evidence(
            Evidence(
                claim_id=claim.id,
                event_id=event.id,
                stance="support" if success else "contradict",
                channel=channel,
                key=independence_key(claim.scope, event),
                user=event.user,
                at=event.at,
                derived=None,
            )
        )
        if claim.status != "deprecated":
            return self._reevaluate(claim, event, ctx, store)
        since = claim.invalid_since
        if not (
            success
            and channel in VERIFIED
            and since is not None
            and event.at > since
        ):
            store.put_claim(claim)  # deprecated claims only collect evidence
            return []
        claim.invalid_since = None  # revived: judged again from its evidence
        out = self._reevaluate(claim, event, ctx, store)
        if claim.status == "deprecated":
            claim.invalid_since = since
            store.put_claim(claim)
        return out

    @staticmethod
    def _retrieved(event: Event, claim_id: str, store: LedgerStore) -> bool:
        """The record repeats what it was handed: it cites the claim, or its
        session was handed the claim before the record was made."""
        if claim_id in event.refs:
            return True
        seen = store.seen(event.session, claim_id)
        return seen is not None and seen < event.at

    @staticmethod
    def _human_fresh(
        evidence: Sequence[Evidence], canonical: str, store: LedgerStore
    ) -> bool:
        """Human support (not derived) at or after the canonical record's time."""
        memory = store.memory(canonical)
        if memory is None:
            return False
        return any(
            e.channel == "human"
            and e.stance == "support"
            and e.derived is None
            and e.at >= memory.at
            for e in evidence
        )

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
        report = evaluate(
            claim,
            counted,
            text,
            self._cfg,
            human_fresh=self._human_fresh(evidence, canonical, store),
        )
        if (
            before in ("established", "disputed")
            and canonical != canonical_before
            and not all(report.checks.values())
        ):
            # A promoted claim's status is sticky, but new text must still
            # pass every gate check: keep the previous canonical and body.
            canonical = canonical_before
            found = ctx.events.get([canonical])
            text = found[0].text if found else ""
            report = evaluate(
                claim,
                counted,
                text,
                self._cfg,
                human_fresh=self._human_fresh(evidence, canonical, store),
            )
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
