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
    return component.replace("%3A", ":").replace("%25", "%")


def _k_mem(event_id: str) -> str:
    return f"mem:{_enc(event_id)}"


def _k_claim(claim_id: str) -> str:
    return f"claim:{_enc(claim_id)}"


def _k_ev(claim_id: str, event_id: str) -> str:
    return f"ev:{_enc(claim_id)}:{_enc(event_id)}"


def _p_ev(claim_id: str) -> str:
    return f"ev:{_enc(claim_id)}:"  # trailing colon: c-1 must not match c-10


def _k_seen(session: str, claim_id: str) -> str:
    return f"seen:{_enc(session)}:{_enc(claim_id)}"


def _k_grp(group_id: str, claim_id: str) -> str:
    return f"grp:{_enc(group_id)}:{_enc(claim_id)}"


def _p_grp(group_id: str) -> str:
    return f"grp:{_enc(group_id)}:"


def _k_sig(scope: Scope, owner: str, signature: str) -> str:
    return f"sig:{_enc(scope)}:{_enc(owner)}:{_enc(signature)}"


class LedgerStore:
    def __init__(self, state: StateStore) -> None:
        self._s = state

    def memory(self, event_id: str) -> Memory | None:
        raw = self._s.get(_k_mem(event_id))
        return None if raw is None else Memory.from_json(raw)

    def put_memory(self, memory: Memory) -> None:
        self._s.put(_k_mem(memory.id), memory.to_json())

    def claim(self, claim_id: str) -> Claim | None:
        raw = self._s.get(_k_claim(claim_id))
        return None if raw is None else Claim.from_json(raw)

    def put_claim(self, claim: Claim) -> None:
        self._s.put(_k_claim(claim.id), claim.to_json())

    def claims(self) -> list[Claim]:
        found = [Claim.from_json(v) for _, v in self._s.scan("claim:")]
        return sorted(found, key=lambda c: c.id)

    def evidence(self, claim_id: str) -> list[Evidence]:
        found = [Evidence.from_json(v) for _, v in self._s.scan(_p_ev(claim_id))]
        return sorted(found, key=lambda e: (e.at, e.event_id))

    def put_evidence(self, evidence: Evidence) -> None:
        self._s.put(_k_ev(evidence.claim_id, evidence.event_id), evidence.to_json())

    def seen(self, session: str, claim_id: str) -> datetime | None:
        raw = self._s.get(_k_seen(session, claim_id))
        return None if raw is None else datetime.fromisoformat(raw)

    def mark_seen(self, session: str, claim_id: str, at: datetime) -> None:
        key = _k_seen(session, claim_id)
        if self._s.get(key) is None:
            self._s.put(key, at.isoformat())

    def claims_in_group(self, group_id: str) -> list[str]:
        prefix = _p_grp(group_id)
        return sorted(_dec(key[len(prefix) :]) for key, _ in self._s.scan(prefix))

    def add_to_group(self, group_id: str, claim_id: str) -> None:
        key = _k_grp(group_id, claim_id)
        if self._s.get(key) is None:
            self._s.put(key, True)

    def claim_by_signature(
        self, scope: Scope, owner: str, signature: str
    ) -> str | None:
        raw = self._s.get(_k_sig(scope, owner, signature))
        return None if raw is None else str(raw)

    def index_signature(
        self, scope: Scope, owner: str, signature: str, claim_id: str
    ) -> None:
        key = _k_sig(scope, owner, signature)
        if self._s.get(key) is None:
            self._s.put(key, claim_id)

    def skip(self, reason: str) -> None:
        counts = self.skipped()
        counts[reason] = counts.get(reason, 0) + 1
        self._s.put(_SKIPPED, counts)

    def skipped(self) -> dict[str, int]:
        raw = self._s.get(_SKIPPED)
        return {} if raw is None else {str(k): int(v) for k, v in raw.items()}
