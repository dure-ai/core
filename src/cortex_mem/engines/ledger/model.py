"""Records the ledger engine keeps in its state store (spec section 5).

Every record converts to and from plain JSON. Datetimes are ISO 8601 strings
with a timezone and come back timezone-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Self, cast

from cortex_mem.contract import Scope

ClaimStatus = Literal[
    "candidate", "provisional", "established", "disputed", "deprecated", "archived"
]
Channel = Literal["observation", "review", "test", "doc", "human"]
Stance = Literal["support", "contradict"]
Derived = Literal["retrieved", "copy"]


def _dt(value: datetime) -> str:
    """ISO 8601 text of a datetime."""
    return value.isoformat()


def _parse(value: str) -> datetime:
    """Parse a stored datetime; a value without a timezone is corrupt and raises ValueError."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"stored datetime has no timezone: {value!r}")
    return parsed


@dataclass(frozen=True)
class Memory:
    """A record event the engine observed, and the claim it joined.

    `id` is the event id, `user`/`team`/`session` its author, `at` its time,
    and `group` the core group the record was in when observed (None without
    embeddings).
    """
    id: str
    claim_id: str
    user: str
    team: str
    session: str
    at: datetime
    group: str | None

    def to_json(self) -> dict[str, Any]:
        """Plain JSON for the state store."""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "user": self.user,
            "team": self.team,
            "session": self.session,
            "at": _dt(self.at),
            "group": self.group,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `to_json` output."""
        return cls(
            id=data["id"],
            claim_id=data["claim_id"],
            user=data["user"],
            team=data["team"],
            session=data["session"],
            at=_parse(data["at"]),
            group=data.get("group"),
        )


@dataclass
class Claim:
    """One claim: the records that say the same thing, and what the gate made of them.

    - `scope`, `owner`: the visibility cell the claim lives in (owner is the
      user at personal scope, the team at team scope).
    - `conditions`: the context the claim holds under; a record with other
      conditions opens another claim.
    - `canonical`: event id of the record whose text is the knowledge body.
    - `first_at`, `last_support_at`: first and last counted support.
    - `invalid_since`: when it was deprecated or archived; only verified
      support after it revives the claim.
    - `promoted`: whether the knowledge base holds it (a Promote went out
      and no Demote since).
    - `use_success`, `use_failure`: reports of using it after it was handed
      over; usefulness, not votes on truth.
    - `report`: the last gate report (`GateReport.to_json`)."""
    id: str
    scope: Scope
    owner: str
    conditions: dict[str, str]
    canonical: str
    status: ClaimStatus
    first_at: datetime
    last_support_at: datetime
    invalid_since: datetime | None
    promoted: bool
    use_success: int
    use_failure: int
    report: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Plain JSON for the state store."""
        return {
            "id": self.id,
            "scope": self.scope,
            "owner": self.owner,
            "conditions": dict(self.conditions),
            "canonical": self.canonical,
            "status": self.status,
            "first_at": _dt(self.first_at),
            "last_support_at": _dt(self.last_support_at),
            "invalid_since": None
            if self.invalid_since is None
            else _dt(self.invalid_since),
            "promoted": self.promoted,
            "use_success": self.use_success,
            "use_failure": self.use_failure,
            "report": self.report,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `to_json` output."""
        since = data.get("invalid_since")
        return cls(
            id=data["id"],
            scope=cast(Scope, data["scope"]),
            owner=data["owner"],
            conditions=dict(data["conditions"]),
            canonical=data["canonical"],
            status=cast(ClaimStatus, data["status"]),
            first_at=_parse(data["first_at"]),
            last_support_at=_parse(data["last_support_at"]),
            invalid_since=None if since is None else _parse(since),
            promoted=data["promoted"],
            use_success=data["use_success"],
            use_failure=data["use_failure"],
            report=data.get("report") or {},
        )


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence on a claim, from one event.

    `stance` is support or contradict, `channel` how it was checked, `key`
    the independence key (events with the same key are one vote), and
    `derived` why it does not count: "retrieved" (restates knowledge the
    session was handed) or "copy" (copies another user's text); None counts."""
    claim_id: str
    event_id: str
    stance: Stance
    channel: Channel
    key: str
    user: str
    at: datetime
    derived: Derived | None

    def to_json(self) -> dict[str, Any]:
        """Plain JSON for the state store."""
        return {
            "claim_id": self.claim_id,
            "event_id": self.event_id,
            "stance": self.stance,
            "channel": self.channel,
            "key": self.key,
            "user": self.user,
            "at": _dt(self.at),
            "derived": self.derived,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `to_json` output."""
        return cls(
            claim_id=data["claim_id"],
            event_id=data["event_id"],
            stance=cast(Stance, data["stance"]),
            channel=cast(Channel, data["channel"]),
            key=data["key"],
            user=data["user"],
            at=_parse(data["at"]),
            derived=cast("Derived | None", data.get("derived")),
        )
