"""Fixed-side store interfaces: what the runtime writes through. Engines never
see these; they get the read-only views in contract.EngineContext.

A durable implementation (SQLite, PostgreSQL) passes the contract when it
passes the suites in cortex_mem.testing.
"""

from collections.abc import Collection, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager
from typing import Any, Protocol

from ..contract import Event, EventReader, Knowledge, KnowledgeReader, Scope


class CursorConflict(Exception):
    """Another process advanced the engine's cursor first."""


class EventStore(EventReader, Protocol):
    """Append-only event log."""

    def append(self, event: Event) -> str:
        """Persist the event and return its id. Raises ValueError on a duplicate id."""
        ...

    def since(self, seq: int) -> Iterator[tuple[int, Event]]:
        """Events appended after position seq, in append order, with their
        positions. Positions start at 1 and only grow; since(0) is everything."""
        ...


class KnowledgeStore(KnowledgeReader, Protocol):
    """The knowledge base. In production the Git wiki is the copy people read."""

    def put(self, item: Knowledge) -> None:
        """Create the item or replace the item with the same id."""
        ...

    def remove(self, id: str) -> None:
        """Remove the item; unknown ids are ignored."""
        ...


class EngineStateBackend(Protocol):
    """One engine's persistent state plus its delivery cursor."""

    def get(self, key: str) -> Any | None: ...

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]: ...

    def cursor(self) -> int:
        """Position of the last event delivered to the engine (0 = none)."""
        ...

    def commit(
        self, expected: int, cursor: int, writes: Mapping[str, Any], deletes: Set[str]
    ) -> None:
        """Apply writes and deletes and move the cursor to `cursor`, all or
        nothing. Raises CursorConflict if the cursor is no longer `expected`."""
        ...

    def version(self) -> str | None:
        """The engine state_version the state was written with (None = never set)."""
        ...

    def reset(self, version: str) -> None:
        """Drop every key, move the cursor back to 0 and record `version`."""
        ...


class VectorBackend(Protocol):
    """Embedding vectors, similarity groups and the embedding cache. Vectors are
    float64 (list[float]); callers get copies. All writes happen inside the
    runtime's transaction()."""

    # Index state
    def meta(self) -> dict | None:
        """{"model": str, "group_threshold": float}, None until reset() is called."""
        ...

    def reset(self, meta: dict) -> None:
        """Drop every vector and group, move the cursor to 0 and record meta.
        The embedding cache stays."""
        ...

    def indexed(self) -> int:
        """Position of the last event indexed (0 = none)."""
        ...

    def set_indexed(self, seq: int) -> None: ...

    # Vectors (kind: "event" | "knowledge")
    def put_vector(self, kind: str, id: str, vector: Sequence[float]) -> None:
        """Store or replace. Replacing keeps the id's original position."""
        ...

    def vector(self, kind: str, id: str) -> list[float] | None: ...

    def vectors(self, kind: str, upto: int | None = None) -> Iterator[tuple[str, list[float]]]:
        """Event vectors in first-insertion order, knowledge vectors by id.
        upto (events only; ValueError for knowledge): just the event vectors
        whose event position, the seq given to put_group, is <= upto."""
        ...

    def delete_vectors(self, kind: str, ids: Collection[str]) -> None:
        """Remove vectors only (not groups); unknown ids are ignored."""
        ...

    # Groups
    def put_group(
        self, event_id: str, group_id: str, scope: Scope, owner: str, seq: int
    ) -> None: ...

    def group_of(self, event_id: str) -> str | None: ...

    def members(self, group_id: str, upto: int | None = None) -> list[str]:
        """Event ids of the group ordered by seq, only those with seq <= upto
        when given; unknown group gives []."""
        ...

    def partition(self, scope: Scope, owner: str) -> list[str]:
        """Grouped event ids of that (scope, owner), ordered by seq."""
        ...

    # Embedding cache
    def cached(self, model: str, digest: str) -> list[float] | None: ...

    def cache(self, model: str, digest: str, vector: Sequence[float]) -> None: ...


class Database(Protocol):
    """All stores behind one connection. The runtime takes them apart:
    Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction)."""

    @property
    def events(self) -> EventStore: ...

    @property
    def knowledge(self) -> KnowledgeStore: ...

    @property
    def vectors(self) -> VectorBackend: ...

    def state(self, engine: str) -> EngineStateBackend:
        """State and cursor of the engine with this name (its import path)."""
        ...

    def transaction(self) -> AbstractContextManager[object]:
        """One write transaction across the stores; may be entered again inside itself."""
        ...
