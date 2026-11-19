"""Behaviour-test helpers for the ledger engine: events, a runtime over an
in-memory database (with or without a similarity port) and state access."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from cortex_mem.contract import EngineContext, Event, Kind, Outcome, Scope, Signal
from cortex_mem.engines.ledger.engine import LedgerEngine
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase
from tests.ledger.fakes import FakeSimilarity

T0 = datetime(2026, 10, 19, tzinfo=UTC)
ENGINE = "ledger"  # name of the engine's state in the database


def ev(
    id: str,
    text: str,
    *,
    kind: Kind = "record",
    user: str = "u1",
    team: str = "t1",
    session: str | None = None,
    signal: Signal | None = "self",
    outcome: Outcome | None = "unknown",
    refs: tuple[str, ...] = (),
    scope: Scope = "team",
    minutes: int = 0,
    conditions: Mapping[str, str] | None = None,
) -> Event:
    """An event at T0 + minutes; without a session, each event has its own."""
    return Event(
        id=id,
        kind=kind,
        at=T0 + timedelta(minutes=minutes),
        user=user,
        team=team,
        session=session if session is not None else f"s-{id}",
        scope=scope,
        text=text,
        signal=signal,
        outcome=outcome,
        refs=refs,
        conditions=dict(conditions or {}),
    )


class _GroupedRuntime(Runtime):
    """A runtime whose engines read a FakeSimilarity port with fixed groups."""

    def __init__(self, *args: Any, groups: Mapping[str, str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._groups = dict(groups)

    def _ctx(self, state, upto):  # type: ignore[override]
        return EngineContext(
            events=self.events,
            knowledge=self.knowledge,
            state=state,
            similar=FakeSimilarity(self.events, self._groups, upto=upto),
        )


def runtime(
    engine: LedgerEngine | None = None,
    db: InMemoryDatabase | None = None,
    groups: Mapping[str, str] | None = None,
) -> tuple[Runtime, InMemoryDatabase]:
    """A runtime over `db`. With `groups` (event id -> label) ctx.similar is a
    FakeSimilarity; without, it is None and the engine takes the lexical path."""
    db = db if db is not None else InMemoryDatabase()
    engine = engine if engine is not None else LedgerEngine()
    args = (engine, db.events, db.knowledge, db.state(ENGINE))
    if groups is None:
        return Runtime(*args, transaction=db.transaction), db
    return _GroupedRuntime(*args, transaction=db.transaction, groups=groups), db


def knowledge_ids(db: InMemoryDatabase) -> list[str]:
    return [k.id for k in db.knowledge.all()]


def state(db: InMemoryDatabase) -> LedgerStore:
    """Read access to the engine's committed state."""
    return LedgerStore(db.state(ENGINE))


def count(db: InMemoryDatabase, prefix: str) -> int:
    """Number of committed state keys under a prefix, e.g. "mem:" or "grp:"."""
    return sum(1 for _ in db.state(ENGINE).scan(prefix))


class _Writable:
    """StateStore over a backend that collects writes for one commit."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.writes: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.writes[key] if key in self.writes else self._backend.get(key)

    def put(self, key: str, value: Any) -> None:
        self.writes[key] = value

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        merged = dict(self._backend.scan(prefix))
        merged.update((k, v) for k, v in self.writes.items() if k.startswith(prefix))
        yield from sorted(merged.items())


def edit_state(db: InMemoryDatabase, change: Callable[[LedgerStore], None]) -> None:
    """Write the engine's state directly (for states a later task produces).
    Call it after the first sync: the runtime resets a state with no version."""
    backend = db.state(ENGINE)
    staged = _Writable(backend)
    change(LedgerStore(staged))
    cursor = backend.cursor()
    backend.commit(cursor, cursor, staged.writes, set(), epoch=backend.epoch())
