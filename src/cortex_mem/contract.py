"""Engine port contract v0.1.

Everything the MCP server, the evaluation harness, and every engine share.
The engine is the only swappable part; the harness sees engines only through
these types. Rules that are not expressible in types (call order, id and scope
semantics) are documented in docs/contract.md and are part of the contract.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

CONTRACT_VERSION = "0.1.0"

Kind = Literal["record", "recall", "outcome", "human_confirm", "commit"]
Signal = Literal["self", "test", "review", "human"]
Outcome = Literal["success", "failure", "unknown"]
Scope = Literal["personal", "team"]
Status = Literal["memory", "knowledge", "invalidated"]


@dataclass(frozen=True)
class Event:
    """One observation. The MCP server and the harness world generator emit
    the same shape; the store keeps them append-only.

    refs by kind:
      recall               -> ids of the Handoffs given to the agent
      outcome/human_confirm -> ids of the events or Handoffs being judged
    """

    id: str
    kind: Kind
    at: datetime  # timezone-aware
    user: str
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

    def __post_init__(self) -> None:
        if self.at.tzinfo is None:
            raise ValueError(f"Event.at must be timezone-aware: {self.id}")


@dataclass(frozen=True)
class QueryContext:
    user: str
    session: str
    scope: Scope
    conditions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Handoff:
    """One item an engine puts into the agent's context."""

    id: str  # stable across calls; recall events carry it in refs
    content: str
    sources: tuple[str, ...]  # event ids, resolvable via EpisodeReader.get
    status: Status | None = None  # shown to the agent, never used for scoring
    conditions: Mapping[str, str] = field(default_factory=dict)
    confidence: float | None = None  # 0..1, enables Brier scoring


class EpisodeReader(Protocol):
    """Read-only view of the store. This is what engines receive."""

    def get(self, ids: Collection[str]) -> list[Event]:
        """Events for the given ids, in the order given; unknown ids are skipped."""
        ...

    def search(
        self,
        query: str,
        ctx: QueryContext,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        """Up to k events visible under ctx, best first (higher score = better).

        personal: only ctx.user's events.
        team: ctx.user's events plus every user's team-scoped events.
        """
        ...


class EpisodeStore(EpisodeReader, Protocol):
    """Append-only store (fixed part, not swapped per engine)."""

    def append(self, event: Event) -> str:
        """Persist the event and return its id. Raises ValueError on a duplicate id."""
        ...


@runtime_checkable
class Engine(Protocol):
    """The engine port. observe() is always called after the event was appended."""

    def observe(self, event: Event) -> None: ...

    def handoff(self, query: str, ctx: QueryContext, k: int = 6) -> list[Handoff]:
        """At most k items, best first. Same event sequence => same result."""
        ...


@runtime_checkable
class KnowledgeSnapshot(Protocol):
    """Optional adapter: what the engine would hand off as of `at`, for
    deterministic list-level scoring. Called after observing every event up to `at`."""

    def snapshot(self, at: datetime) -> list[Handoff]: ...


EngineFactory = Callable[[EpisodeReader], Engine]
