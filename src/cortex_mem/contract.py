"""Engine port contract v0.1.

Everything the MCP server, the evaluation harness, and every engine share.
The engine is the only swappable part. It reads the fixed stores through
EngineContext and returns decisions; the runtime (runtime.py) validates them
and writes the knowledge base. Rules that are not expressible in types (call
order, id and scope semantics) are documented in docs/contract.md and are
part of the contract.
"""

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

CONTRACT_VERSION = "0.1.0"

Kind = Literal["record", "recall", "outcome", "human_confirm", "commit"]
Signal = Literal["self", "test", "review", "human", "doc"]
Outcome = Literal["success", "failure", "unknown"]
Scope = Literal["personal", "team", "org"]
Status = Literal["memory", "knowledge", "invalidated"]
KnowledgeKind = Literal["knowledge", "skill"]
DemoteWhy = Literal["invalid", "stale"]


def _freeze(obj: object, name: str) -> None:
    """Replace a mapping field with a read-only copy, so a value handed to an
    engine can never change what a store holds."""
    object.__setattr__(obj, name, MappingProxyType(dict(getattr(obj, name))))


def _require_tz(at: datetime, what: str) -> None:
    if at.tzinfo is None:
        raise ValueError(f"{what} must be timezone-aware")


@dataclass(frozen=True)
class Event:
    """One observation. The MCP server and the harness world generator emit
    the same shape; the store keeps them append-only.

    refs by kind:
      recall               -> ids of the Handoffs given to the agent
      outcome/human_confirm -> ids of the events or Handoffs being judged;
                               must not be empty (Runtime.append refuses it)

    A recall event's repo and conditions are where the question was asked
    (Caller.repo and Caller.context).

    text_redacted: the caller left out the text it was given. The server does
    this for an outcome whose audience is wider than the repo's policy: the
    judgement (outcome, signal, refs) is shared, the agent's words are not.
    """

    id: str
    kind: Kind
    at: datetime  # timezone-aware
    user: str
    team: str
    session: str
    scope: Scope
    text: str = ""
    agent: str | None = None
    signal: Signal | None = None
    outcome: Outcome | None = None
    conditions: Mapping[str, str] = field(default_factory=dict)  # when the claim holds
    refs: tuple[str, ...] = ()
    repo: str | None = None  # where the evidence came from, e.g. commit or CI run
    rev: str | None = None
    text_redacted: bool = False

    def __post_init__(self) -> None:
        _require_tz(self.at, f"Event.at ({self.id})")
        _freeze(self, "conditions")


@dataclass(frozen=True)
class Caller:
    """Who is asking, and from where. The server fills it from the connection
    and its config, the harness from the scenario (virtual time). repo and
    context let an engine tell where a conditional fact applies; matching
    them against Event.conditions is the engine's job."""

    user: str
    team: str
    session: str
    at: datetime  # timezone-aware
    agent: str | None = None
    repo: str | None = None  # the repository being worked in
    context: Mapping[str, str] = field(default_factory=dict)  # e.g. {"service": "api"}; keys as in conditions

    def __post_init__(self) -> None:
        _require_tz(self.at, "Caller.at")
        _freeze(self, "context")


@dataclass(frozen=True)
class Knowledge:
    """One promoted item: a page in the knowledge base (the Git wiki)."""

    id: str  # chosen by the engine, stable: it is the wiki path
    kind: KnowledgeKind
    title: str
    body: str  # markdown
    scope: Scope
    owner: str  # personal: user id, team: team id, org: "*" (one org in v0.1)
    sources: tuple[str, ...]  # event ids; must not be empty
    conditions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze(self, "conditions")


@dataclass(frozen=True)
class Promote:
    """Create the item, or replace the item with the same id (an update).
    A skill is promoted the same way, with item.kind = "skill"."""

    item: Knowledge


@dataclass(frozen=True)
class Demote:
    """Remove an item from the knowledge base. The wiki history keeps it, so
    the item was valid from its Promote commit until `since`.

    why: "invalid" = it was or became false; "stale" = no longer relevant
    (archived). The runtime handles both alike; the harness counts them apart."""

    id: str
    reason: str
    sources: tuple[str, ...]  # event ids that justify it, e.g. failure outcomes
    why: DemoteWhy = "invalid"
    since: datetime | None = None  # timezone-aware; when it stopped holding, if known

    def __post_init__(self) -> None:
        if self.since is not None:
            _require_tz(self.since, f"Demote.since ({self.id})")


Decision = Promote | Demote


@dataclass(frozen=True)
class Handoff:
    """One item an engine puts into the agent's context."""

    id: str  # stable across calls; recall events carry it in refs
    content: str
    sources: tuple[str, ...]  # event ids; the runtime drops items whose sources the caller cannot see
    status: Status | None = None  # shown to the agent, never used for scoring
    conditions: Mapping[str, str] = field(default_factory=dict)
    confidence: float | None = None  # 0..1, enables Brier scoring

    def __post_init__(self) -> None:
        _freeze(self, "conditions")


class EventReader(Protocol):
    """Read-only view of the event log."""

    def get(self, ids: Collection[str]) -> list[Event]:
        """Events for the given ids, in the order given; unknown ids are skipped."""
        ...

    def search(
        self,
        query: str,
        caller: Caller,
        scope: Scope,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        """Up to k events visible to caller under scope, best first (higher score = better).
        Visibility is access.event_visible."""
        ...


class KnowledgeReader(Protocol):
    """Read-only view of the knowledge base as it stands now."""

    def get(self, ids: Collection[str]) -> list[Knowledge]:
        """Items for the given ids, in the order given; unknown ids are skipped."""
        ...

    def all(self) -> list[Knowledge]:
        """Every item, ordered by id."""
        ...

    def search(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        """Up to k items visible to caller under scope, best first.
        Visibility is access.knowledge_visible."""
        ...


class StateStore(Protocol):
    """The engine's own persistent key-value store. Values are JSON
    (dict, list, str, int, float, bool, None) and are copied on the way in
    and out. Writes made during observe or tick become durable only if the
    call returns; in handoff the store is read-only."""

    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any) -> None: ...

    def delete(self, key: str) -> None: ...

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Keys starting with prefix, in key order."""
        ...


@dataclass(frozen=True)
class EngineContext:
    """What the fixed part hands an engine on every call."""

    events: EventReader
    knowledge: KnowledgeReader
    state: StateStore


@runtime_checkable
class Engine(Protocol):
    """The engine port. Keep durable state in ctx.state, not on the instance:
    several server processes may run the same engine against the same stores.

    An engine may set a class attribute `state_version: str`. When it differs
    from the version the stored state was written with, the runtime drops the
    state and delivers every event again from the start of the log."""

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        """Called once per event, in append order, after the event was appended."""
        ...

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        """Time passed. Expire, demote or clean up what depends on time."""
        ...

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        """At most k items, best first. Same event sequence => same result."""
        ...


EngineFactory = Callable[[Mapping[str, str]], Engine]  # engine config only; stores come through ctx
