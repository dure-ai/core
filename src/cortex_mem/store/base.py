"""Fixed-side store interfaces: what the runtime writes through. Engines never
see these; they get the read-only views in contract.EngineContext.

A durable implementation (SQLite, PostgreSQL) passes the contract when it
passes the suites in cortex_mem.testing.
"""

from collections.abc import Collection, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import AbstractContextManager
from typing import Any, Protocol

from ..contract import (
    Caller,
    Event,
    EventReader,
    Kind,
    Knowledge,
    KnowledgeReader,
    Scope,
)


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

    def clear(self) -> None:
        """Remove every item at once (an engine reset empties the knowledge base).
        The owner stays."""
        ...

    def owner(self) -> str | None:
        """Which engine state the knowledge base belongs to, as the runtime
        recorded it with set_owner (None = never recorded)."""
        ...

    def set_owner(self, owner: str) -> None:
        """Record the owner. Durable stores keep it with the knowledge base,
        and a transaction that rolls back undoes it."""
        ...


class EngineStateBackend(Protocol):
    """One engine's persistent state plus its delivery cursor."""

    def get(self, key: str) -> Any | None: ...

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]: ...

    def cursor(self) -> int:
        """Position of the last event delivered to the engine (0 = none)."""
        ...

    def epoch(self) -> int:
        """Counts every commit and every reset of this engine (0 = neither yet).
        Read it before reading what a commit is based on, and pass it to commit()."""
        ...

    def commit(
        self,
        expected: int,
        cursor: int,
        writes: Mapping[str, Any],
        deletes: AbstractSet[str],
        *,
        epoch: int,
    ) -> None:
        """Apply writes and deletes and move the cursor to `cursor`, all or
        nothing, and move the epoch on to `epoch` + 1. Raises CursorConflict,
        writing nothing, if the cursor is no longer `expected` or the epoch is
        no longer `epoch`: the cursor alone cannot tell two commits from the
        same cursor apart (a tick keeps it) or a commit from before a reset
        (which puts it back to 0)."""
        ...

    def version(self) -> str | None:
        """The engine state_version the state was written with (None = never set)."""
        ...

    def position(self) -> tuple[int, int, str | None]:
        """(epoch(), cursor(), version()) read together, in one step on a
        durable store: what a delivery reads before it starts and checks
        before it commits, at the cost of one round trip."""
        ...

    def reset(self, version: str) -> None:
        """Drop every key, move the cursor back to 0, record `version` and
        add 1 to the epoch."""
        ...


class VectorBackend(Protocol):
    """Embedding vectors, similarity groups and the embedding cache, and the
    similarity searches over them. Vectors are float32: every write (vectors
    and cache) rounds to float32, and reads give those values back as Python
    floats (list[float]); callers get copies. All writes happen inside the
    runtime's transaction().

    The searches judge visibility from the events and knowledge of the same
    Database (cortex_mem.access rules), so a backend is built next to them.
    An event's position here is the seq given to put_group: only grouped
    event vectors are search candidates, upto bounds that seq, and ties go
    to the smaller seq (then the smaller id)."""

    kind: str
    """The store this backend lives in: "memory", "sqlite" or "postgres".
    Part of the index version, so moving to another store rebuilds the index."""

    # Index state
    def meta(self) -> dict[str, Any] | None:
        """{"model": str, "group_threshold": float, "store": str} as the Index
        passed it to reset(), None until reset() is called."""
        ...

    def reset(self, meta: dict[str, Any]) -> None:
        """Drop every vector (and the knowledge digests) and group, move the cursor
        to 0, record meta and add 1 to generation(), all in one step. The embedding
        cache stays."""
        ...

    def generation(self) -> int:
        """Which index this is: 0 until the first reset(), then one more for each
        reset(). Not part of meta(). Durable stores keep it across reopening."""
        ...

    def indexed(self) -> int:
        """Position of the last event indexed (0 = none)."""
        ...

    def position(self) -> tuple[dict[str, Any] | None, int, int]:
        """(meta(), generation(), indexed()) read together, in one step on a
        durable store, so the three come from one moment."""
        ...

    def set_indexed(self, expected: int, seq: int, generation: int) -> None:
        """Move the position from `expected` to seq, atomically. Raises
        CursorConflict, writing nothing, if indexed() is not `expected` or
        generation() is not `generation`. Raised inside a caller's transaction
        on a durable store, the conflict rolls back that whole transaction."""
        ...

    # Vectors (kind: "event" | "knowledge")
    def put_vector(
        self, kind: str, id: str, vector: Sequence[float], digest: str | None = None
    ) -> None:
        """Store or replace. Replacing keeps the id's original position. For
        "knowledge", digest (of the text the vector was made from) is stored with
        it, and replacing replaces it (None leaves the id without one); for
        "event" it is ignored."""
        ...

    def knowledge_digests(self, ids: Collection[str] | None = None) -> dict[str, str]:
        """id -> digest of the knowledge vectors that have one; with `ids`, of
        those ids only (unknown ids are left out)."""
        ...

    def vector(self, kind: str, id: str) -> list[float] | None:
        """The stored vector of that kind and id, or None."""
        ...

    def vectors(self, kind: str, upto: int | None = None) -> Iterator[tuple[str, list[float]]]:
        """Event vectors in first-insertion order, knowledge vectors by id.
        upto (events only; ValueError for knowledge): just the event vectors
        whose event position, the seq given to put_group, is <= upto."""
        ...

    def delete_vectors(self, kind: str, ids: Collection[str]) -> None:
        """Remove vectors (and knowledge digests) only, not groups; unknown ids are ignored."""
        ...

    # Groups
    def put_group(
        self, event_id: str, group_id: str, scope: Scope, owner: str, seq: int
    ) -> None:
        """Put the event in group_id within the (scope, owner) cell at position
        seq. An event has one group: putting it again moves it, with its new
        cell and seq."""
        ...

    def group_of(self, event_id: str) -> str | None:
        """The event's group id, or None when it is not grouped."""
        ...

    def members(self, group_id: str, upto: int | None = None) -> list[str]:
        """Event ids of the group ordered by seq, only those with seq <= upto
        when given; unknown group gives []."""
        ...

    def partition(self, scope: Scope, owner: str) -> list[str]:
        """Grouped event ids of that (scope, owner), ordered by seq."""
        ...

    # Embedding cache
    def cached(self, model: str, digest: str) -> list[float] | None:
        """The cached vector of the text with this digest embedded by this model,
        or None. Keyed by model, so models never mix; reset() keeps the cache."""
        ...

    def cache(self, model: str, digest: str, vector: Sequence[float]) -> None:
        """Store or replace the vector of the text with this digest embedded by
        this model (float32, any dimension)."""
        ...

    # Similarity: cosine of the float32 query against the stored vectors.
    def nearest_member(
        self, scope: Scope, owner: str, vector: Sequence[float], upto: int
    ) -> tuple[str, float] | None:
        """The grouped event of the (scope, owner) cell with seq <= upto whose
        vector is closest to `vector`, with its score, whatever the score;
        ties go to the smaller seq. None when the cell has no such event.
        Always exact (never an approximate index): groups must not vary."""
        ...

    def search_events(
        self,
        vector: Sequence[float],
        caller: Caller,
        scope: Scope,
        k: int,
        kinds: Collection[Kind] | None = None,
        upto: int | None = None,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Up to k (event id, score) of the grouped events the caller may see
        in scope, with score > 0, best first, ties by seq. kinds limits the
        event kinds, upto the seq (None = no bound), exclude drops one id."""
        ...

    def search_knowledge(
        self, vector: Sequence[float], caller: Caller, scope: Scope, k: int
    ) -> list[tuple[str, float]]:
        """Up to k (knowledge id, score) of the items the caller may see in
        scope, with score > 0, best first, ties by id."""
        ...


class Database(Protocol):
    """All stores behind one connection. The runtime takes them apart:
    Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction)."""

    @property
    def events(self) -> EventStore:
        """The event log."""
        ...

    @property
    def knowledge(self) -> KnowledgeStore:
        """The knowledge base."""
        ...

    @property
    def vectors(self) -> VectorBackend:
        """The vectors, groups and embedding cache of the similarity index."""
        ...

    def state(self, engine: str) -> EngineStateBackend:
        """State and cursor of the engine with this name (its import path)."""
        ...

    def transaction(self) -> AbstractContextManager[object]:
        """One write transaction across the stores; may be entered again inside itself."""
        ...
