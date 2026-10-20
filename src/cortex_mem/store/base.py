"""Fixed-side store interfaces: what the runtime writes through. Engines never
see these; they get the read-only views in contract.EngineContext.

A durable implementation (SQLite, PostgreSQL) passes the contract when it
passes tests/test_store.py.
"""

from collections.abc import Iterator, Mapping, Set
from typing import Any, Protocol

from ..contract import Event, EventReader, Knowledge, KnowledgeReader


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
