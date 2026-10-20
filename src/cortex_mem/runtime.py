"""The fixed runtime between callers (MCP server, harness) and an engine.

It is the only writer: it appends events, delivers each to the engine exactly
once, checks every decision before it reaches the knowledge base, and filters
what the engine hands off. The server and the harness run this same code, so
a harness score describes what the server does.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .access import audience_sees, event_visible, knowledge_visible
from .contract import (
    Caller,
    Decision,
    Demote,
    Engine,
    EngineContext,
    Event,
    Handoff,
    Knowledge,
    Promote,
    Scope,
)
from .store.base import CursorConflict, EngineStateBackend, EventStore, KnowledgeStore

_ABSENT = object()
MAX_STATE_VALUE_BYTES = 64 * 1024  # per key, JSON-encoded


@dataclass(frozen=True)
class Verdict:
    """What the runtime did with one decision: the decision log. The harness
    reports it as an observed metric; it does not count toward the verdict."""

    decision: object
    accepted: bool
    reason: str | None  # why it was refused; None when accepted
    at: datetime


class _StagedState:
    """StateStore handed to the engine: writes stay here until the runtime commits."""

    def __init__(self, backend: EngineStateBackend, read_only: bool = False) -> None:
        self._backend = backend
        self._read_only = read_only
        self.writes: dict[str, Any] = {}
        self.deletes: set[str] = set()

    def get(self, key: str) -> Any | None:
        if key in self.deletes:
            return None
        if key in self.writes:
            return self.writes[key]
        return self._backend.get(key)

    def put(self, key: str, value: Any) -> None:
        self._check_writable()
        size = len(json.dumps(value).encode())
        if size > MAX_STATE_VALUE_BYTES:
            raise ValueError(f"state value for {key!r} is {size} bytes; split it across keys")
        self.deletes.discard(key)
        self.writes[key] = value

    def delete(self, key: str) -> None:
        self._check_writable()
        self.writes.pop(key, None)
        self.deletes.add(key)

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        merged = dict(self._backend.scan(prefix))
        merged.update((k, v) for k, v in self.writes.items() if k.startswith(prefix))
        for key in sorted(merged):
            if key not in self.deletes:
                yield key, merged[key]

    def _check_writable(self) -> None:
        if self._read_only:
            raise PermissionError("state is read-only during handoff")


class Runtime:
    """`transaction` opens one write transaction across the stores. A durable
    backend passes it so that delivering one event (cursor, engine state and
    knowledge writes) is all or nothing and serialized across processes."""

    def __init__(
        self,
        engine: Engine,
        events: EventStore,
        knowledge: KnowledgeStore,
        state: EngineStateBackend,
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        self.engine = engine
        self.events = events
        self.knowledge = knowledge
        self.state = state
        self.transaction = transaction
        self.verdicts: list[Verdict] = []
        self._version_checked = False

    @property
    def rejections(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]

    def _ctx(self, state: _StagedState) -> EngineContext:
        return EngineContext(events=self.events, knowledge=self.knowledge, state=state)

    def append(self, event: Event) -> str:
        """Append an event and deliver everything not yet delivered."""
        self.events.append(event)
        self.sync()
        return event.id

    def sync(self) -> int:
        """Deliver events appended since the cursor, in order, one transaction
        each. Returns how many this call delivered. If the engine raises, that
        event stays undelivered and is retried next time."""
        self._check_state_version()
        delivered = 0
        while True:
            with self.transaction():
                cursor = self.state.cursor()
                pending = next(iter(self.events.since(cursor)), None)
                if pending is None:
                    return delivered
                seq, event = pending
                staged = _StagedState(self.state)
                decisions = self.engine.observe(event, self._ctx(staged))
                try:
                    self._commit(decisions, staged, cursor, seq, event.at)
                except CursorConflict:
                    continue  # another process delivered it first; read the cursor again
            delivered += 1

    def _check_state_version(self) -> None:
        """Drop the engine's state and start delivery over when the engine's
        state_version differs from the stored one. The knowledge base is kept;
        replayed Promotes replace items with the same id."""
        want = getattr(self.engine, "state_version", None)
        if want is None or self._version_checked:
            return
        with self.transaction():
            if self.state.version() != want:
                self.state.reset(want)
        self._version_checked = True

    def tick(self, now: datetime) -> None:
        self.sync()
        with self.transaction():
            cursor = self.state.cursor()
            staged = _StagedState(self.state)
            decisions = self.engine.tick(now, self._ctx(staged))
            try:
                self._commit(decisions, staged, cursor, cursor, now)
            except CursorConflict:
                return  # events arrived meanwhile; the next tick sees them

    def handoff(self, query: str, caller: Caller, scope: Scope, k: int = 6) -> list[Handoff]:
        self.sync()
        staged = _StagedState(self.state, read_only=True)
        handoffs = self.engine.handoff(query, caller, scope, self._ctx(staged), k)
        return [h for h in handoffs if self._may_see(h, caller, scope)][:k]

    def recall(
        self, query: str, caller: Caller, scope: Scope, event_id: str, k: int = 6
    ) -> list[Handoff]:
        """handoff, then record what was handed off as a recall event (refs = ids).
        The recall event is personal whatever scope was searched: a query says
        what someone is working on, and is not shared by searching widely."""
        handoffs = self.handoff(query, caller, scope, k)
        self.append(
            Event(
                id=event_id, kind="recall", at=caller.at, user=caller.user, team=caller.team,
                session=caller.session, scope="personal", text=query, agent=caller.agent,
                refs=tuple(h.id for h in handoffs),
            )
        )
        return handoffs

    def _commit(
        self,
        decisions: Sequence[Decision],
        staged: _StagedState,
        expected: int,
        cursor: int,
        at: datetime,
    ) -> None:
        pending: dict[str, Knowledge | None] = {}  # id -> item, or None when demoted
        accepted: list[Decision] = []
        verdicts: list[Verdict] = []
        for decision in decisions:
            reason = self._check(decision, pending)
            verdicts.append(Verdict(decision, reason is None, reason, at))
            if reason is not None:
                continue
            accepted.append(decision)
            if isinstance(decision, Promote):
                pending[decision.item.id] = decision.item
            else:
                pending[decision.id] = None
        self.state.commit(expected, cursor, staged.writes, staged.deletes)
        self.verdicts.extend(verdicts)
        for decision in accepted:
            if isinstance(decision, Promote):
                self.knowledge.put(decision.item)
            else:
                self.knowledge.remove(decision.id)

    def _current(self, id: str, pending: dict[str, Knowledge | None]) -> Knowledge | None:
        found = pending.get(id, _ABSENT)
        if found is not _ABSENT:
            return found  # type: ignore[return-value]
        items = self.knowledge.get([id])
        return items[0] if items else None

    def _check(self, decision: object, pending: dict[str, Knowledge | None]) -> str | None:
        if isinstance(decision, Promote):
            item = decision.item
            if not item.id:
                return "empty id"
            return self._check_sources(item.sources, item.scope, item.owner)
        if isinstance(decision, Demote):
            current = self._current(decision.id, pending)
            if current is None:
                return f"no knowledge item {decision.id!r}"
            return self._check_sources(decision.sources, current.scope, current.owner)
        return f"not a decision: {type(decision).__name__}"

    def _check_sources(self, sources: tuple[str, ...], scope: Scope, owner: str) -> str | None:
        if not sources:
            return "no sources"
        found = {e.id: e for e in self.events.get(sources)}
        missing = [s for s in sources if s not in found]
        if missing:
            return f"unknown sources: {missing}"
        hidden = [e.id for e in found.values() if not audience_sees(e, scope, owner)]
        if hidden:
            return f"sources not visible to the {scope} audience {owner!r}: {hidden}"
        return None

    def _may_see(self, handoff: Handoff, caller: Caller, scope: Scope) -> bool:
        if not handoff.sources:
            return False
        found = self.events.get(handoff.sources)
        if {e.id for e in found} != set(handoff.sources):
            return False
        if not all(event_visible(e, caller, scope) for e in found):
            return False
        items = self.knowledge.get([handoff.id])
        return not items or knowledge_visible(items[0], caller, scope)
