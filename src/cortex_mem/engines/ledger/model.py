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
    return value.isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"stored datetime has no timezone: {value!r}")
    return parsed


@dataclass(frozen=True)
class Memory:
    id: str
    claim_id: str
    user: str
    team: str
    session: str
    at: datetime
    group: str | None

    def to_json(self) -> dict[str, Any]:
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
    claim_id: str
    event_id: str
    stance: Stance
    channel: Channel
    key: str
    user: str
    at: datetime
    derived: Derived | None

    def to_json(self) -> dict[str, Any]:
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
