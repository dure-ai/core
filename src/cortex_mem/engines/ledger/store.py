"""Typed access to the ledger engine's state (spec section 5).

All key strings are built by the ``_k_*`` functions below and nowhere else.
"""

from __future__ import annotations

from datetime import datetime

from cortex_mem.contract import Scope, StateStore
from cortex_mem.engines.ledger.model import Claim, Evidence, Memory

_SKIPPED = "ledger:skipped"


def _enc(component: str) -> str:
    """Percent-encode '%' and ':' so no component holds the raw separator."""
    return component.replace("%", "%25").replace(":", "%3A")


def _dec(component: str) -> str:
    """Undo `_enc`."""
    return component.replace("%3A", ":").replace("%25", "%")


def _k_mem(event_id: str) -> str:
    """`mem:<event>`: the memory a record became."""
    return f"mem:{_enc(event_id)}"


def _k_claim(claim_id: str) -> str:
    """`claim:<claim>`: one claim."""
    return f"claim:{_enc(claim_id)}"


def _k_ev(claim_id: str, event_id: str) -> str:
    """`ev:<claim>:<event>`: one evidence row; the event id makes redelivery idempotent."""
    return f"ev:{_enc(claim_id)}:{_enc(event_id)}"


def _p_ev(claim_id: str) -> str:
    """Scan prefix of a claim's evidence rows."""
    return f"ev:{_enc(claim_id)}:"  # trailing colon: c-1 must not match c-10


def _k_seen(session: str, claim_id: str) -> str:
    """`seen:<session>:<claim>`: when the session was first handed the claim."""
    return f"seen:{_enc(session)}:{_enc(claim_id)}"


def _k_judged(claim_id: str, event_id: str) -> str:
    """`judged:<claim>:<event>`: a judgment event already applied to the claim."""
    return f"judged:{_enc(claim_id)}:{_enc(event_id)}"


def _k_grp(group_id: str, claim_id: str) -> str:
    """`grp:<group>:<claim>`: a claim that holds records of a core group."""
    return f"grp:{_enc(group_id)}:{_enc(claim_id)}"


def _p_grp(group_id: str) -> str:
    """Scan prefix of the claims of a core group."""
    return f"grp:{_enc(group_id)}:"


def _k_skip(reason: str, event_id: str, detail: str) -> str:
    """`skip:<reason>:<event>:<detail>`: a skip already counted, so a redelivered
    event is not counted again (`detail` names the ref or claim, or is empty)."""
    return f"skip:{_enc(reason)}:{_enc(event_id)}:{_enc(detail)}"


def _k_sig(scope: Scope, owner: str, signature: str) -> str:
    """`sig:<scope>:<owner>:<signature>`: the claim a text fingerprint
    (`text.signature`) points at."""
    return f"sig:{_enc(scope)}:{_enc(owner)}:{_enc(signature)}"


class LedgerStore:
    """Typed reads and writes over the engine's `ctx.state`.

    Made for one observe, tick or handoff call. Reads go to the state store,
    so they see the writes staged by the current delivery, except a claim's
    evidence: it is read once and kept, and `put_evidence` updates the kept
    rows as it writes. Every evidence write goes through this class, so the
    kept rows stay what the state store holds."""

    def __init__(self, state: StateStore) -> None:
        """Wrap `state` (the engine's `ctx.state`)."""
        self._s = state
        self._evidence: dict[str, dict[str, Evidence]] = {}  # claim -> event -> row

    def memory(self, event_id: str) -> Memory | None:
        """The memory a record event became, or None if it was never observed."""
        raw = self._s.get(_k_mem(event_id))
        return None if raw is None else Memory.from_json(raw)

    def put_memory(self, memory: Memory) -> None:
        """Write a memory under its event id."""
        self._s.put(_k_mem(memory.id), memory.to_json())

    def claim(self, claim_id: str) -> Claim | None:
        """One claim, or None."""
        raw = self._s.get(_k_claim(claim_id))
        return None if raw is None else Claim.from_json(raw)

    def put_claim(self, claim: Claim) -> None:
        """Write a claim, replacing the stored one."""
        self._s.put(_k_claim(claim.id), claim.to_json())

    def claims(self) -> list[Claim]:
        """Every claim, in id order (tick decides claims in this order)."""
        found = [Claim.from_json(v) for _, v in self._s.scan("claim:")]
        return sorted(found, key=lambda c: c.id)

    def evidence(self, claim_id: str) -> list[Evidence]:
        """The claim's evidence rows, oldest first (ties by event id)."""
        rows = self._evidence.get(claim_id)
        if rows is None:
            found = (Evidence.from_json(v) for _, v in self._s.scan(_p_ev(claim_id)))
            rows = self._evidence[claim_id] = {e.event_id: e for e in found}
        return sorted(rows.values(), key=lambda e: (e.at, e.event_id))

    def put_evidence(self, evidence: Evidence) -> None:
        """Write an evidence row; the same claim and event overwrite one row."""
        self._s.put(_k_ev(evidence.claim_id, evidence.event_id), evidence.to_json())
        rows = self._evidence.get(evidence.claim_id)
        if rows is not None:
            rows[evidence.event_id] = evidence

    def seen(self, session: str, claim_id: str) -> datetime | None:
        """When the session was first handed the claim, or None."""
        raw = self._s.get(_k_seen(session, claim_id))
        return None if raw is None else datetime.fromisoformat(raw)

    def mark_seen(self, session: str, claim_id: str, at: datetime) -> None:
        """Record that the session was handed the claim; keeps the first time."""
        key = _k_seen(session, claim_id)
        if self._s.get(key) is None:
            self._s.put(key, at.isoformat())

    def judged(self, claim_id: str, event_id: str) -> bool:
        """Whether this judgment event was already applied to the claim."""
        return self._s.get(_k_judged(claim_id, event_id)) is not None

    def mark_judged(self, claim_id: str, event_id: str) -> None:
        """Mark the judgment event as applied, so redelivery does not count it again."""
        self._s.put(_k_judged(claim_id, event_id), True)

    def claims_in_group(self, group_id: str) -> list[str]:
        """Ids of the claims holding records of a core group, sorted."""
        prefix = _p_grp(group_id)
        return sorted(_dec(key[len(prefix) :]) for key, _ in self._s.scan(prefix))

    def add_to_group(self, group_id: str, claim_id: str) -> None:
        """Note that the claim holds a record of the core group."""
        key = _k_grp(group_id, claim_id)
        if self._s.get(key) is None:
            self._s.put(key, True)

    def claim_by_signature(
        self, scope: Scope, owner: str, signature: str
    ) -> str | None:
        """The claim a text fingerprint (`text.signature`) points at in this
        (scope, owner), or None."""
        raw = self._s.get(_k_sig(scope, owner, signature))
        return None if raw is None else str(raw)

    def index_signature(
        self,
        scope: Scope,
        owner: str,
        signature: str,
        claim_id: str,
        *,
        replace: bool = False,
    ) -> None:
        """Point the signature at the claim: only if it has none, or always
        with `replace`."""
        key = _k_sig(scope, owner, signature)
        if replace or self._s.get(key) is None:
            self._s.put(key, claim_id)

    def skip(self, reason: str, event_id: str, detail: str = "") -> None:
        """Count one event the engine passed over, under `reason`
        (`org_scope`, `unknown_outcome`, `unknown_ref`, `out_of_scope_evidence`,
        `verified_not_canonical`).
        `detail` tells apart several skips of one event (the unknown ref, the
        claim out of scope). The same (reason, event, detail) counts once, so
        observing an event again does not change the counts."""
        marker = _k_skip(reason, event_id, detail)
        if self._s.get(marker) is not None:
            return
        self._s.put(marker, True)
        counts = self.skipped()
        counts[reason] = counts.get(reason, 0) + 1
        self._s.put(_SKIPPED, counts)

    def skipped(self) -> dict[str, int]:
        """Skip counts by reason."""
        raw = self._s.get(_SKIPPED)
        return {} if raw is None else {str(k): int(v) for k, v in raw.items()}
