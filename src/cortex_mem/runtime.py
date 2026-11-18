"""The fixed runtime between callers (MCP server, harness) and an engine.

It is the only writer: it appends events, delivers each to the engine exactly
once, checks every decision before it reaches the knowledge base, and filters
what the engine hands off. The server and the harness run this same code, so
a harness score describes what the server does.
"""

import json
import sys
from collections.abc import Callable, Collection, Iterator, Sequence
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
from .embed import EmbeddingError
from .index import Index, IndexConfigChanged
from .store.base import CursorConflict, EngineStateBackend, EventStore, KnowledgeStore

_ABSENT = object()
MAX_CONFIG_RETRIES = 3  # per event: another config rebuilt the index; rebuild ours and try again
MAX_STATE_VALUE_BYTES = 64 * 1024  # per key, JSON-encoded


@dataclass(frozen=True)
class Verdict:
    """What the runtime did with one decision: the decision log. The harness
    reports it as an observed metric; it does not count toward the verdict."""

    decision: object
    accepted: bool
    reason: str | None  # why it was refused; None when accepted
    at: datetime


@dataclass(frozen=True)
class Skipped:
    """An event the runtime stopped trying to deliver after `max_failures`
    failed attempts. The engine never saw it; the event stays in the log."""

    seq: int
    event_id: str
    attempts: int
    error: str  # the last exception, as "Type: message"


class _Skip(Exception):
    """Raised out of the delivery transaction for an event given up on, so
    that transaction ends (rolled back) before the skip commits in a new one."""

    def __init__(
        self, cursor: int, epoch: int, seq: int, event_id: str, attempts: int, error: Exception
    ) -> None:
        super().__init__(event_id)
        self.cursor = cursor
        self.epoch = epoch
        self.seq = seq
        self.event_id = event_id
        self.attempts = attempts
        self.error = error


class _StagedState:
    """StateStore handed to the engine: writes stay here until the runtime commits.

    Pending values are kept as JSON text, so a value is copied on the way in
    and on the way out, as it is once committed."""

    def __init__(self, backend: EngineStateBackend, read_only: bool = False) -> None:
        self._backend = backend
        self._read_only = read_only
        self._writes: dict[str, str] = {}
        self.deletes: set[str] = set()

    @property
    def writes(self) -> dict[str, Any]:
        """Pending writes as fresh values, for the backend commit."""
        return {key: json.loads(raw) for key, raw in self._writes.items()}

    def get(self, key: str) -> Any | None:
        if key in self.deletes:
            return None
        if key in self._writes:
            return json.loads(self._writes[key])
        return self._backend.get(key)

    def put(self, key: str, value: Any) -> None:
        self._check_writable()
        raw = json.dumps(value)
        size = len(raw.encode())
        if size > MAX_STATE_VALUE_BYTES:
            raise ValueError(f"state value for {key!r} is {size} bytes; split it across keys")
        self.deletes.discard(key)
        self._writes[key] = raw

    def delete(self, key: str) -> None:
        self._check_writable()
        self._writes.pop(key, None)
        self.deletes.add(key)

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        merged = dict(self._backend.scan(prefix))
        merged.update((k, json.loads(v)) for k, v in self._writes.items() if k.startswith(prefix))
        for key in sorted(merged):
            if key not in self.deletes:
                yield key, merged[key]

    def _check_writable(self) -> None:
        if self._read_only:
            raise PermissionError("state is read-only during handoff")


class Runtime:
    """`transaction` opens one write transaction across the stores. A durable
    backend passes it so that delivering one event (cursor, engine state and
    knowledge writes) is all or nothing and serialized across processes.

    `max_failures` is how many times `observe` may fail on one event before the
    runtime skips it and moves on. None (the default) never skips: the error is
    raised every time, and delivery stays stuck on that event. A long-running
    server sets it so one bad event cannot block every caller; the harness
    leaves it None so an engine failure is never hidden. Failures are counted
    per Runtime (per process), and skipped events are listed in `skipped`.

    `index` turns on embeddings. Engines read the index through
    `ctx.similar`; None (the default) leaves `ctx.similar` None. Indexing is a
    step of its own that runs before each delivery transaction, never inside
    one: the index embeds with no write transaction open and writes in short
    transactions of its own, so a slow embedding API never holds a durable
    store's write lock. An event is delivered only once the index has reached
    it. An embedding failure is raised and the event is not delivered; it is
    retried next time and never counted toward `max_failures`.

    Knowledge vectors follow the knowledge base after a decision commits: the
    ids that accepted Promotes and Demotes touched are synced once the
    delivery transaction has ended. If that embedding fails the decision
    stands, the error is raised, and the ids are tried again first on this
    Runtime's next sync; the first sync of each Runtime (and the one after
    the index is rebuilt) compares every knowledge item, which catches up on
    a process that died in between.

    `observe` and `tick` still run inside the delivery transaction, so an
    engine query there (`ctx.similar.search` and `ctx.similar.knowledge`
    embed an uncached query text) is embedded under the write lock, each call
    bounded by the embedder's timeout. Engines should prefer `group` and
    `neighbors` in `observe`; they read only stored vectors and never embed.
    """

    def __init__(
        self,
        engine: Engine,
        events: EventStore,
        knowledge: KnowledgeStore,
        state: EngineStateBackend,
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
        max_failures: int | None = None,
        index: Index | None = None,
    ) -> None:
        if max_failures is not None and max_failures < 1:
            raise ValueError("max_failures must be at least 1, or None to never skip")
        self.engine = engine
        self.events = events
        self.knowledge = knowledge
        self.state = state
        self.transaction = transaction
        self.max_failures = max_failures
        self.index = index
        self.verdicts: list[Verdict] = []
        self.skipped: list[Skipped] = []
        self._failures: dict[int, int] = {}  # seq -> failed observe attempts in this process
        self._pending_knowledge: set[str] = set()  # ids whose vectors failed to follow the knowledge base
        self._knowledge_reconciled = False  # every knowledge item compared once by this Runtime

    @property
    def rejections(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]

    def _staged(self, read_only: bool = False) -> _StagedState:
        return _StagedState(self.state, read_only=read_only)

    def _ctx(self, state: _StagedState, upto: int) -> EngineContext:
        """upto: the last event the engine has been given (the one it observes,
        else its cursor); ctx.similar reads no event after it."""
        similar = None if self.index is None else self.index.reader(self.events, self.knowledge, upto)
        return EngineContext(events=self.events, knowledge=self.knowledge, state=state, similar=similar)

    def append(self, event: Event) -> str:
        """Append an event and deliver everything not yet delivered. An outcome
        or human_confirm must say what it judges (non-empty refs)."""
        if event.kind in ("outcome", "human_confirm") and not event.refs:
            raise ValueError(f"{event.kind} event {event.id!r} needs refs: the ids it judges")
        self.events.append(event)
        self.sync()
        return event.id

    def sync(self) -> int:
        """Deliver events appended since the cursor, in order, one transaction
        each. Returns how many this call delivered. If the engine raises, that
        event stays undelivered and is retried next time, until `max_failures`
        attempts have failed; then it is skipped (see `skipped`). With an
        index, each event is indexed before its delivery transaction, and an
        EmbeddingError while indexing it (or while embedding knowledge) is
        raised as is: the event stays undelivered and is never skipped for it.
        If another process rebuilds the index with another config meanwhile,
        it is rebuilt for this one and delivery starts again, at most
        MAX_CONFIG_RETRIES times per event; then IndexConfigChanged is raised."""
        self._check_versions()
        self._follow_knowledge()
        delivered = 0
        ahead = self.index is not None
        outage: tuple[int, EmbeddingError] | None = None  # (indexed position, error) of the batched pass
        retries: dict[int, int] = {}  # seq -> rebuilds after another config took the index
        while True:
            # Read without a lock: the delivery transaction reads both again.
            cursor = self.state.cursor()
            pending = next(iter(self.events.since(cursor)), None)
            if pending is None:
                return delivered
            seq = pending[0]
            if self.index is not None:
                try:
                    if ahead:
                        # Index everything appended so far in a few batched embed calls.
                        # A failure is kept for the event whose text failed: the events
                        # before it are still delivered, and it is not embedded again.
                        ahead = False
                        try:
                            self.index.advance(self.events, sys.maxsize)
                        except EmbeddingError as exc:
                            outage = self.index.indexed(), exc
                    if outage is not None and seq > outage[0] and self.index.indexed() < seq:
                        raise outage[1]
                    # Not the engine's failure: raised before the observe try.
                    self.index.advance(self.events, seq)
                except IndexConfigChanged:
                    # Another process rebuilt the index with another config. Rebuild
                    # it for ours (a new generation: that replays this engine) and
                    # try again, a few times: two configs on one store are not supported.
                    retries[seq] = retries.get(seq, 0) + 1
                    if retries[seq] > MAX_CONFIG_RETRIES:
                        raise
                    self._check_versions()
                    self._follow_knowledge()
                    ahead, outage = True, None
                    continue
            outcome, touched = self._deliver_next(seq)
            if outcome is None:
                return delivered
            delivered += outcome
            self._sync_knowledge(touched)

    def _deliver_next(self, seq: int) -> tuple[int | None, list[str]]:
        """Deliver the next event in one transaction, if it is the one at `seq`
        (the one indexed for it): (1, ids its accepted decisions touched) if
        delivered, (0, []) if not (skipped, or another process moved the
        cursor or the index first: read again), (None, []) if nothing is
        pending."""
        try:
            with self.transaction():
                epoch = self.state.epoch()  # before the cursor and the state the engine reads
                cursor = self.state.cursor()
                pending = next(iter(self.events.since(cursor)), None)
                if pending is None:
                    return None, []
                if pending[0] != seq:
                    return 0, []  # another process delivered meanwhile
                if self.index is not None and (self.index.stale() or self.index.indexed() < seq):
                    return 0, []  # another process rebuilt the index meanwhile: index again first
                event = pending[1]
                staged = self._staged()
                try:
                    decisions = self.engine.observe(event, self._ctx(staged, seq))
                except Exception as exc:
                    attempts = self._give_up(seq)
                    if attempts is None:
                        raise
                    # Leave the transaction by raising, so a durable store rolls it back:
                    # after a failed statement PostgreSQL accepts nothing more in it.
                    raise _Skip(cursor, epoch, seq, event.id, attempts, exc) from exc
                try:
                    touched = self._commit(decisions, staged, cursor, epoch, seq, event.at)
                except CursorConflict:
                    return 0, []  # another process delivered it first; read the cursor again
                self._failures.pop(seq, None)
        except _Skip as skip:
            self._skip(skip)
            return 0, []
        return 1, touched

    def _skip(self, skip: "_Skip") -> None:
        """Move the cursor past a given-up event in a new transaction, with none
        of its writes: the engine's were staged and are dropped, and nothing
        else in the failed transaction is committed by this one. The event is
        reported in `skipped` only once this has committed; if it fails, the
        error is raised and the event is given up again on the next attempt.
        The index is not touched: it reached this event before delivery."""
        try:
            with self.transaction():
                self.state.commit(skip.cursor, skip.seq, {}, set(), epoch=skip.epoch)
        except CursorConflict:
            self._failures.pop(skip.seq, None)  # another process moved on first
            return
        self._failures.pop(skip.seq, None)
        error = f"{type(skip.error).__name__}: {skip.error}"
        self.skipped.append(Skipped(skip.seq, skip.event_id, skip.attempts, error))

    def _sync_knowledge(self, touched: Collection[str]) -> None:
        """After a commit, outside its transaction: let the knowledge vectors
        follow the ids the decisions touched. On an EmbeddingError the
        decisions stand; the ids wait for this Runtime's next sync."""
        if self.index is None or not touched:
            return
        try:
            self.index.sync_knowledge(self.knowledge, touched)
        except EmbeddingError:
            self._pending_knowledge.update(touched)
            raise

    def _follow_knowledge(self) -> None:
        """At the start of a sync: compare every knowledge item on this
        Runtime's first sync (and after a rebuild), else retry the ids whose
        embedding failed. An EmbeddingError is raised."""
        if self.index is None:
            return
        if not self._knowledge_reconciled:
            self.index.sync_knowledge(self.knowledge, None)
            self._knowledge_reconciled = True
            self._pending_knowledge.clear()
        elif self._pending_knowledge:
            ids = set(self._pending_knowledge)
            self.index.sync_knowledge(self.knowledge, ids)
            self._pending_knowledge -= ids

    def _give_up(self, seq: int) -> int | None:
        """Count a failed attempt; the attempt count when the event is to be
        skipped now, else None. The count is kept until the skip commits."""
        attempts = self._failures.get(seq, 0) + 1
        self._failures[seq] = attempts
        if self.max_failures is None or attempts < self.max_failures:
            return None
        return attempts

    def _want_version(self) -> str | None:
        """The version this engine's state must have been written with. With
        an index it is the JSON of [state_version or None, index fingerprint]:
        the state records which index generation it was built on. Without
        one it is the engine's state_version, and None means nothing to check."""
        state_version = getattr(self.engine, "state_version", None)
        if self.index is None:
            return state_version
        return json.dumps([state_version, self.index.fingerprint()], sort_keys=True, separators=(",", ":"))

    def _check_versions(self) -> None:
        """Runs on every sync. With an index, first rebuild it if it is stale
        (Index.ensure_current, which embeds with no transaction open). Then,
        when the stored version differs from _want_version(), drop the
        engine's state and start delivery over, in one short transaction that
        computes the version again (the index may have been rebuilt
        meanwhile). A rebuild by any process adds 1 to the index generation,
        so it changes the version of every engine on the store. The knowledge
        base is kept; replayed Promotes replace items with the same id."""
        if self.index is not None and self.index.ensure_current(self.knowledge):
            self._knowledge_reconciled = False  # compare every item again on this sync
        want = self._want_version()
        if want is None or self.state.version() == want:
            return  # the common case takes no lock
        with self.transaction():
            want = self._want_version()
            if want is not None and self.state.version() != want:
                self.state.reset(want)

    def tick(self, now: datetime) -> None:
        self.sync()
        with self.transaction():
            epoch = self.state.epoch()
            cursor = self.state.cursor()
            staged = self._staged()
            decisions = self.engine.tick(now, self._ctx(staged, cursor))
            try:
                touched = self._commit(decisions, staged, cursor, epoch, cursor, now)
            except CursorConflict:
                return  # events arrived meanwhile; the next tick sees them
        self._sync_knowledge(touched)

    def handoff(self, query: str, caller: Caller, scope: Scope, k: int = 6) -> list[Handoff]:
        self.sync()
        staged = self._staged(read_only=True)
        handoffs = self.engine.handoff(query, caller, scope, self._ctx(staged, self.state.cursor()), k)
        return [h for h in handoffs if self._may_see(h, caller, scope)][:k]

    def recall(
        self, query: str, caller: Caller, scope: Scope, event_id: str, k: int = 6
    ) -> list[Handoff]:
        """handoff, then record what was handed off as a recall event (refs = ids,
        repo and conditions = the caller's repo and context). The recall event
        is personal whatever scope was searched: a query says what someone is
        working on, and is not shared by searching widely."""
        handoffs = self.handoff(query, caller, scope, k)
        self.append(
            Event(
                id=event_id, kind="recall", at=caller.at, user=caller.user, team=caller.team,
                session=caller.session, scope="personal", text=query, agent=caller.agent,
                refs=tuple(h.id for h in handoffs), repo=caller.repo, conditions=caller.context,
            )
        )
        return handoffs

    def _commit(
        self,
        decisions: Sequence[Decision],
        staged: _StagedState,
        expected: int,
        epoch: int,
        cursor: int,
        at: datetime,
    ) -> list[str]:
        """Inside the delivery transaction: check the decisions, commit the
        engine's state and cursor, apply the accepted decisions. Returns the
        knowledge ids they touched, for the index to follow afterwards."""
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
        self.state.commit(expected, cursor, staged.writes, staged.deletes, epoch=epoch)
        self.verdicts.extend(verdicts)
        for decision in accepted:
            if isinstance(decision, Promote):
                self.knowledge.put(decision.item)
            else:
                self.knowledge.remove(decision.id)
        return list(pending)

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
