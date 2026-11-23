"""LedgerEngine: records become claims, claims collect evidence, and claims
that pass the gate become knowledge (spec sections 4, 6 and 9).

The engine keeps nothing on the instance but its configuration: everything
durable is in ctx.state, so observing one event twice (contract 0.4.0) has
no effect beyond the state and the decisions returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import ClassVar

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
from cortex_mem.engines.ledger.matcher import DOWN, eligible, match_claim
from cortex_mem.engines.ledger.model import Claim, Derived, Evidence, Memory
from cortex_mem.engines.ledger.similar import similar_of
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.engines.ledger.text import signature


class LedgerEngine:
    """Evidence-ledger engine for the core engine port (`contract.Engine`).

    Groups records into claims, keeps one vote per independent source,
    promotes a claim when it passes the gate and demotes it on verified
    contradiction or long silence. Load it on the server with
    `CORTEX_ENGINE=cortex_mem.engines.ledger:LedgerEngine`."""

    STATE_FORMAT: ClassVar[str] = "2"  # 2: evidence rows carry `record`

    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        """Parse `config` (string keys and values, see README); None or an
        empty mapping means the defaults. Raises ValueError on an unknown
        key or an out-of-range value."""
        self._cfg = LedgerConfig.from_mapping(config)

    @property
    def state_version(self) -> str:
        """The state format and a digest of the settings. The settings decide
        what becomes knowledge, so a store delivered under other settings is
        replayed under these instead of continued (contract: state_version)."""
        return f"{self.STATE_FORMAT}+{self._cfg.fingerprint()}"

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        """Apply one event and return the decisions it causes.

        A record joins or opens a claim and supports it; an outcome or
        human_confirm judges the claims its refs name; a recall marks the
        claims the session was handed. Only the claims the event touches are
        re-evaluated. Other kinds are ignored."""
        store = LedgerStore(ctx.state)
        if event.kind == "record":
            return self._record(event, ctx, store)
        if event.kind in ("outcome", "human_confirm"):
            return self._judge(event, ctx, store)
        if event.kind == "recall":
            for claim_id in self._resolve(event, store, count_unknown=False):
                store.mark_seen(event.session, claim_id, event.at)
        return []

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        """Spec section 10: when `review_after` is set, archive each established
        claim whose last support is older than it, and demote its knowledge.
        The archival time goes to `invalid_since`: like a deprecated claim, an
        archived one only collects evidence until verified support dated
        after it revives the claim. Claims are decided in id order; the
        result is idempotent because an archived claim is no longer
        established."""
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
            claim.invalid_since = now  # only verified support after it revives
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
        """At most k items for the query, one per claim, best first, each
        with its canonical text, a provenance line and all its counted
        support as sources. Reads only (see `retrieve.handoff`)."""
        store = LedgerStore(ctx.state)
        return retrieve.handoff(query, caller, scope, ctx, store, self._cfg, k)

    def _record(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Spec section 6, `record`: find or open the claim, write the memory
        and its support evidence, then re-evaluate the claim."""
        if event.scope == "org":
            store.skip("org_scope", event.id)  # v0.1 covers personal and team only
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
            self._index_signature(event, claim, store)

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
                record=True,
            )
        )
        return self._reevaluate(claim, event, ctx, store)

    @staticmethod
    def _index_signature(event: Event, claim: Claim, store: LedgerStore) -> None:
        """Index the record's signature. A signature that points at a claim
        this record could not join (taken down, or other conditions) is
        taken over by the record's claim, so the next restatement joins it
        instead of opening yet another claim (spec 7)."""
        sig = signature(event.text)
        if sig is None:
            return  # no words to sign: matching falls back to search
        current = store.claim_by_signature(claim.scope, claim.owner, sig)
        stale = (
            current is not None
            and current != claim.id
            and not eligible(store.claim(current), event)
        )
        store.index_signature(claim.scope, claim.owner, sig, claim.id, replace=stale)

    @staticmethod
    def _resolve(
        event: Event, store: LedgerStore, *, count_unknown: bool = True
    ) -> list[str]:
        """Claim ids the event's refs name: a claim id as it is, an event id
        through its memory. Unique, in order; unknown refs are counted."""
        found: dict[str, None] = {}
        for ref in event.refs:
            if store.claim(ref) is not None:
                found[ref] = None
            elif (memory := store.memory(ref)) is not None:
                found[memory.claim_id] = None
            elif count_unknown:
                store.skip("unknown_ref", event.id, ref)
        return list(found)

    def _judge(
        self, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Spec section 6, `outcome` and `human_confirm`: add support or
        contradiction to the claims the refs name, when in scope."""
        if event.scope == "org":
            # v0.1 covers personal and team only: an org judgment would let
            # other teams vote and hide the claim from team-scope handoffs
            store.skip("org_scope", event.id)
            return []
        if event.outcome not in ("success", "failure"):
            store.skip("unknown_outcome", event.id)
            return []
        out: list[Decision] = []
        for claim_id in self._resolve(event, store):
            claim = store.claim(claim_id)
            assert claim is not None
            if not in_scope(event, claim):
                store.skip("out_of_scope_evidence", event.id, claim_id)
                continue
            if store.judged(claim_id, event.id):
                continue  # delivered again: this judgment is already counted
            store.mark_judged(claim_id, event.id)
            out += self._judge_claim(claim, event, ctx, store)
        return out

    def _judge_claim(
        self, claim: Claim, event: Event, ctx: EngineContext, store: LedgerStore
    ) -> list[Decision]:
        """Apply one judgment event to one claim.

        An agent's own success report counts as use (`use_success`), not as
        evidence, and only if the session was handed the claim at or before
        it. Every other judgment becomes a support or contradiction row (an
        own failure report also counts `use_failure`). A live claim is then
        re-evaluated; a deprecated or archived one only collects the row,
        unless the row is verified support dated after `invalid_since`,
        which revives it for re-evaluation."""
        success = event.outcome == "success"
        own_report = event.kind == "outcome" and event.signal in (None, "self")
        if own_report and success:
            # usefulness, not truth: no vote; only once the session was handed it
            seen = store.seen(event.session, claim.id)
            if seen is not None and seen <= event.at:
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
                record=False,
            )
        )
        down = claim.status
        if down not in DOWN:  # taken down: collect evidence, do not re-judge
            return self._reevaluate(claim, event, ctx, store)
        since = claim.invalid_since
        if not (
            success
            and channel in VERIFIED
            and since is not None
            and event.at > since
        ):
            store.put_claim(claim)  # down claims only collect evidence
            return []
        claim.invalid_since = None  # revived: judged again from its evidence
        out = self._reevaluate(claim, event, ctx, store)
        if claim.status == down:
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
    def _human_fresh(evidence: Sequence[Evidence], canonical: str) -> bool:
        """Human support (not derived) at or after the canonical record's time."""
        made = next((e.at for e in evidence if e.record and e.event_id == canonical), None)
        if made is None:
            return False
        return any(
            e.channel == "human"
            and e.stance == "support"
            and e.derived is None
            and e.at >= made
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
        records = {e.event_id for e in evidence if e.record}
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
            human_fresh=self._human_fresh(evidence, canonical),
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
                human_fresh=self._human_fresh(evidence, canonical),
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
