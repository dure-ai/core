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
from typing import Any, cast

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


class StateVersionChanged(Exception):
    """Another process kept resetting this engine's state to another version
    while this one delivered (two versions of one engine on one store, which
    is not supported): raised after MAX_CONFIG_RETRIES restarts of one event."""


class _VersionMoved(Exception):
    """A commit found the engine's state under another version than the one
    this Runtime builds: check the versions again and start the event over."""


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
    backend passes it so that committing one event's result (cursor, engine
    state and knowledge writes) is all or nothing and serialized across
    processes. The engine itself runs outside it (see below).

    `max_failures` is how many times `observe` may fail on one event before the
    runtime skips it and moves on. None (the default) never skips: the error is
    raised every time, and delivery stays stuck on that event. A long-running
    server sets it so one bad event cannot block every caller; the harness
    leaves it None so an engine failure is never hidden. Failures are counted
    per Runtime (per process), and skipped events are listed in `skipped`.

    `index` turns on embeddings. Engines read the index through
    `ctx.similar`; None (the default) leaves `ctx.similar` None. Indexing is a
    step of its own that runs before each event is observed, never inside a
    write transaction: the index embeds with no write transaction open and writes in short
    transactions of its own, so a slow embedding API never holds a durable
    store's write lock. An event is delivered only once the index has reached
    it. An embedding failure is raised and the event is not delivered; it is
    retried next time and never counted toward `max_failures`.

    Knowledge vectors follow the knowledge base after a decision commits: the
    ids that accepted Promotes and Demotes touched are synced once the
    commit transaction has ended. If that embedding fails the decision
    stands, the error is raised, and the ids are tried again first on this
    Runtime's next sync; the first sync of each Runtime (and the one after
    the index is rebuilt) compares every knowledge item, which catches up on
    a process that died in between.

    `observe` and `tick` run with no write transaction open (contract 0.4.0):
    the engine reads the event log, its state and the index while nothing is
    locked, and its result is committed in one short transaction that first
    checks that nothing it read has moved: the engine's epoch and cursor
    (a commit, a skip or a reset by any process moves the epoch), and with an
    index, the index generation read before observe and an index position at
    or past the event, and that the engine's state is still under the version
    this Runtime expects (the engine's state_version, with the index
    fingerprint when there is an index). When any of them moved, the result
    is dropped and the event is read and observed again; nothing is written
    and no failure is counted. So an engine may observe one event, or tick at
    one cursor, more than once, even in two processes at once, and only one
    result lands. A slow engine or embedder never holds a durable store's
    write lock.

    The restart for a moved index or engine version is bounded: after
    MAX_CONFIG_RETRIES (3) restarts of one event, sync raises
    IndexConfigChanged or StateVersionChanged. Processes with another index
    config, or another engine state_version, on one store are not supported;
    they fail loudly instead of mixing.

    `sync()` returns how many events it delivered; `pending()` returns how
    many still wait after the engine's cursor.
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
        """A fresh staged view of the engine's state for one call; read_only
        raises PermissionError on a write (handoff)."""
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
        """Deliver events appended since the cursor, in order. Returns how many
        this call delivered (see pending() for how many still wait). Each event
        is read with no transaction open, observed with none open, and
        committed in a short checked transaction (_deliver_next). If the
        engine raises, that event stays undelivered and is retried next time,
        until `max_failures` attempts have failed; then it is skipped (see
        `skipped`). A failure is counted only if the engine's epoch did not
        move while it observed. With an index, each event is indexed before
        it is observed, and an EmbeddingError while indexing it (or while
        embedding knowledge, or raised inside observe) is raised as is: the
        event stays undelivered and is never skipped for it.

        If another process rebuilds the index, or resets this engine under
        another state version, the versions are checked again (rebuilding the
        index for this config and dropping the engine's state when it was
        built on another) and the event starts over. An IndexConfigChanged
        raised inside observe takes the same path. After MAX_CONFIG_RETRIES
        such restarts of one event, IndexConfigChanged (index) or
        StateVersionChanged (engine version) is raised; neither is counted
        toward max_failures."""
        self._check_versions()
        self._follow_knowledge()
        done = self.state.cursor()  # delivered or skipped, here or by another process
        self._failures = {seq: n for seq, n in self._failures.items() if seq > done}
        delivered = 0
        ahead = self.index is not None
        outage: tuple[int, EmbeddingError] | None = None  # (indexed position, error) of the batched pass
        retries: dict[int, int] = {}  # seq -> restarts after the index or the engine's version moved
        while True:
            # No transaction: the commit checks the epoch and cursor read here,
            # both in one read, before the state the engine reads.
            epoch, cursor, _ = self.state.position()
            pending = next(iter(self.events.since(cursor)), None)
            if pending is None:
                return delivered
            seq, event = pending
            generation = None
            try:
                if self.index is not None:
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
                    # Not the engine's failure: raised before observe runs.
                    self.index.advance(self.events, seq)
                    generation = self.index.generation()
                outcome, touched = self._deliver_next(epoch, cursor, seq, event, generation)
            except (IndexConfigChanged, _VersionMoved) as exc:
                # Another process rebuilt the index (maybe with another config) or
                # reset this engine under another version. Rebuild the index for
                # ours if needed, check the engine's version again (a reset replays
                # it) and start the event over, a few times: two configs or two
                # engine versions on one store are not supported.
                retries[seq] = retries.get(seq, 0) + 1
                if retries[seq] > MAX_CONFIG_RETRIES:
                    if isinstance(exc, _VersionMoved):
                        raise StateVersionChanged(
                            f"the state of this engine keeps moving to another version: {self.state.version()!r}"
                        ) from None
                    raise
                self._check_versions()
                self._follow_knowledge()
                ahead, outage = self.index is not None, None
                continue
            delivered += outcome
            self._sync_knowledge(touched)

    def pending(self) -> int:
        """How many events wait after the engine's cursor (read now, no lock)."""
        return sum(1 for _ in self.events.since(self.state.cursor()))

    def waiting(self) -> list[str]:
        """Ids of the events that wait after the engine's cursor, in order
        (read now, no lock): an event is delivered (or skipped) once it is
        not among them."""
        return [event.id for _, event in self.events.since(self.state.cursor())]

    def _deliver_next(
        self, epoch: int, cursor: int, seq: int, event: Event, generation: int | None
    ) -> tuple[int, list[str]]:
        """Observe the event at `seq` with no write transaction open, then
        commit the result in one short transaction: (1, ids its accepted
        decisions touched) when it landed, (0, []) when it did not (skipped,
        or something it read moved: the caller reads again and starts over).

        `epoch` and `cursor` were read before the event, and `generation` (with
        an index) after indexing it. The commit first checks that the epoch,
        the cursor and the index generation are still those, that the index
        has reached `seq`, and that the engine's state is still under the
        version this Runtime builds on that generation (else _VersionMoved is
        raised: the caller checks the versions again), and only then checks
        the decisions, commits the
        engine's state and cursor (state.commit checks the epoch and cursor
        once more, atomically, and is the one place that raises
        CursorConflict) and applies the accepted decisions. A failure of the
        engine is counted toward max_failures; nothing else is: a store error
        while committing, an EmbeddingError or IndexConfigChanged is raised."""
        staged = self._staged()
        try:
            # Materialized here: a generator would run engine code inside the commit.
            decisions = list(self.engine.observe(event, self._ctx(staged, seq)))
        except (EmbeddingError, IndexConfigChanged):
            raise  # not the engine's failure: retried, never counted
        except Exception as exc:
            if self.state.epoch() != epoch:
                return 0, []  # reset or moved on meanwhile: not a failure on this state; read again
            attempts = self._give_up(seq)
            if attempts is None:
                raise
            self._skip(cursor, seq, epoch, event.id, attempts, exc)
            return 0, []
        try:
            with self.transaction():
                moved = self._moved(epoch, cursor, generation, seq)
                if moved is None:
                    touched, verdicts = self._commit(decisions, staged, cursor, epoch, seq, event.at)
        except CursorConflict:
            return 0, []  # another process delivered, skipped or reset first; read again
        if moved == "version":
            raise _VersionMoved(seq)
        if moved is not None:
            return 0, []
        self.verdicts.extend(verdicts)
        self._failures.pop(seq, None)
        return 1, touched

    def _moved(self, epoch: int, cursor: int, generation: int | None, seq: int) -> str | None:
        """Inside the commit transaction, before anything is checked or written:
        None when what the engine read is still current; "read" when the
        epoch, the cursor, the index generation or the index position moved
        (read again and start over); "version" when the engine's state is not
        under the version this Runtime builds on `generation`, such as after a
        reset by another version of the engine, or a rebuild of the index
        since the version was checked (check the versions again first). The
        epoch and cursor are checked again by state.commit; this read spares
        the decision checks of a result that is going to be dropped."""
        now_epoch, now_cursor, version = self.state.position()
        if now_epoch != epoch or now_cursor != cursor:
            return "read"
        if self.index is not None:
            now_generation, indexed = self.index.position()
            if now_generation != generation or indexed < seq:
                return "read"
        want = self._want(generation)
        if want is not None and version != want:
            return "version"
        return None

    def _skip(self, cursor: int, seq: int, epoch: int, event_id: str, attempts: int, error: Exception) -> None:
        """Move the cursor past a given-up event in one short transaction, with
        none of its writes (the engine's were staged and are dropped). The
        event is reported in `skipped` only once this has committed; if it
        fails, the error is raised and the event is given up again on the next
        attempt. The index is not touched: it reached this event before
        observe ran."""
        try:
            with self.transaction():
                self.state.commit(cursor, seq, {}, set(), epoch=epoch)
        except CursorConflict:
            # Another process delivered or skipped it, or the engine was reset.
            # An epoch-only conflict (a tick or a reset, the event still pending)
            # also starts the count again: correct, only slower to skip.
            self._failures.pop(seq, None)
            return
        self._failures.pop(seq, None)
        self.skipped.append(Skipped(seq, event_id, attempts, f"{type(error).__name__}: {error}"))

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
        return self._want(None)

    def _want(self, generation: int | None) -> str | None:
        """_want_version() for an index generation the caller has read (None:
        read it now)."""
        state_version = getattr(self.engine, "state_version", None)
        if self.index is None:
            return state_version
        fingerprint = self.index.fingerprint(generation)
        return json.dumps([state_version, fingerprint], sort_keys=True, separators=(",", ":"))

    def _check_versions(self) -> None:
        """Runs on every sync, and again when a commit finds the index rebuilt
        or the engine's state under another version. With an index, first
        rebuild it if it is stale (Index.ensure_current, which embeds with no
        transaction open). Then, when the stored version differs from
        _want_version(), drop the engine's state and start delivery over, in
        one short transaction that computes the version again (the index may
        have been rebuilt meanwhile). A rebuild by any process adds 1 to the
        index generation, so it changes the version of every engine on the
        store. The same transaction empties the knowledge base: the replay
        promotes again what the engine still holds true, and an item it no
        longer promotes would otherwise stay with no engine state left to
        demote it. Knowledge vectors follow on this sync."""
        if self.index is not None and self.index.ensure_current(self.knowledge):
            self._knowledge_reconciled = False  # compare every item again on this sync
        want = self._want_version()
        if want is None or self.state.version() == want:
            return  # the common case takes no lock
        with self.transaction():
            want = self._want_version()
            if want is not None and self.state.version() != want:
                self.state.reset(want)
                self.knowledge.clear()
                self._knowledge_reconciled = False  # drop the vectors of what was removed

    def tick(self, now: datetime) -> None:
        """Deliver what is pending, then let time pass: engine.tick runs with
        no write transaction open, on the state at the cursor, and its result
        is committed in one short transaction with the checks of a delivery
        (the cursor stays). If the epoch, the cursor, the index generation or
        the engine's version moved meanwhile (another process delivered,
        ticked or reset, or the index was rebuilt), the result is dropped; the
        next tick runs again. A tick that decides nothing and writes no state
        commits nothing: it takes no write lock and leaves the epoch alone,
        so it never refuses another process's observe."""
        self.sync()
        epoch, cursor, _ = self.state.position()  # one read, before the state the engine reads
        generation = None if self.index is None else self.index.generation()
        staged = self._staged()
        decisions = list(self.engine.tick(now, self._ctx(staged, cursor)))
        if not decisions and not staged.writes and not staged.deletes:
            return
        try:
            with self.transaction():
                if self._moved(epoch, cursor, generation, cursor) is not None:
                    return
                touched, verdicts = self._commit(decisions, staged, cursor, epoch, cursor, now)
        except CursorConflict:
            return  # another process delivered, ticked or reset first
        self.verdicts.extend(verdicts)
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
    ) -> tuple[list[str], list[Verdict]]:
        """Inside the short commit transaction, after _moved: check the
        decisions, commit the engine's state and cursor (state.commit raises
        CursorConflict if the epoch or cursor moved), apply the accepted
        decisions. Returns the knowledge ids they touched, for the index to
        follow afterwards, and the verdicts, which the caller logs once the
        transaction has committed."""
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
        for decision in accepted:
            if isinstance(decision, Promote):
                self.knowledge.put(decision.item)
            else:
                self.knowledge.remove(decision.id)
        return list(pending), verdicts

    def _current(self, id: str, pending: dict[str, Knowledge | None]) -> Knowledge | None:
        found = pending.get(id, _ABSENT)
        if found is not _ABSENT:
            return cast("Knowledge | None", found)
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
