"""In-memory stores for tests and harness runs. BM25 over the visible documents,
with the SQLite store's tokens; ties keep append order for events and id order
for knowledge, so results are deterministic."""

import bisect
import json
import math
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ..access import event_visible, knowledge_visible
from ..contract import (
    Caller,
    Event,
    EventReader,
    Kind,
    Knowledge,
    KnowledgeReader,
    Scope,
)
from ..embed import QueryScorer, float32, top_by_cosine
from .base import CursorConflict
from .text import bigrams, words

# The harness's all_store baseline uses these, so a score gap between it and an
# engine on this store comes from the engine rather than from the ranking formula.
_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    return words(text) + bigrams(text)


def _bm25(query: str, texts: Sequence[str]) -> list[float]:
    """Score each text against the query; `texts` is the whole corpus for IDF."""
    # IDF counts only the texts passed in (visible to the caller, of the requested
    # kinds), like the harness all_store. SQLite FTS5 counts every row of the table
    # and uses k1=1.2 with an unshifted IDF, so the two stores rank differently;
    # harness comparisons use this store only (contract §4).
    # Every search re-tokenizes the corpus: harness scenarios are small, so this
    # stays simple on purpose instead of keeping an index.
    docs =[Counter(_tokens(text)) for text in texts]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / len(docs) if docs else 0.0
    scores = [0.0] * len(docs)
    # Sorted so the float sums do not depend on the hash seed.
    for term in sorted(set(_tokens(query))):
        df = sum(1 for doc in docs if term in doc)
        if df == 0:
            continue
        idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(docs):
            f = doc[term]
            if f:
                norm = 1 - _B + _B * lengths[i] / average
                scores[i] += idf * f * (_K1 + 1) / (f + _K1 * norm)
    return scores


def _copy(value: Any) -> Any:
    """JSON round trip: rejects what a durable store could not hold, and
    keeps callers from sharing mutable objects with the store."""
    return json.loads(json.dumps(value))


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: dict[str, Event] = {}
        self._log: list[Event] = []

    def append(self, event: Event) -> str:
        if event.id in self._events:
            raise ValueError(f"duplicate event id: {event.id}")
        self._events[event.id] = event
        self._log.append(event)
        return event.id

    def since(self, seq: int) -> Iterator[tuple[int, Event]]:
        for i in range(seq, len(self._log)):
            yield i + 1, self._log[i]

    def count_since(self, seq: int) -> int:
        return max(0, len(self._log) - seq)

    def get(self, ids: Collection[str]) -> list[Event]:
        return [self._events[i] for i in ids if i in self._events]

    def search(
        self,
        query: str,
        caller: Caller,
        scope: Scope,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        visible = [
            e for e in self._log
            if (kinds is None or e.kind in kinds) and event_visible(e, caller, scope)
        ]
        scores = _bm25(query, [e.text for e in visible])
        scored = [(e, score) for e, score in zip(visible, scores) if score > 0]
        scored.sort(key=lambda pair: -pair[1])  # stable: ties keep insertion order
        return scored[:k]


class InMemoryKnowledgeBase:
    def __init__(self) -> None:
        self._items: dict[str, Knowledge] = {}
        self._owner: str | None = None

    def put(self, item: Knowledge) -> None:
        self._items[item.id] = item

    def remove(self, id: str) -> None:
        self._items.pop(id, None)

    def clear(self) -> None:
        self._items.clear()

    def owner(self) -> str | None:
        return self._owner

    def set_owner(self, owner: str) -> None:
        self._owner = owner

    def get(self, ids: Collection[str]) -> list[Knowledge]:
        return [self._items[i] for i in ids if i in self._items]

    def all(self) -> list[Knowledge]:
        return [self._items[i] for i in sorted(self._items)]

    def search(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        visible = [item for item in self.all() if knowledge_visible(item, caller, scope)]
        scores = _bm25(query, [f"{item.title} {item.body}" for item in visible])
        scored = [(item, score) for item, score in zip(visible, scores) if score > 0]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]


class InMemoryStateBackend:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}  # stored as JSON text, like a durable store
        self._cursor = 0
        self._epoch = 0
        self._version: str | None = None

    def get(self, key: str) -> Any | None:
        raw = self._data.get(key)
        return None if raw is None else json.loads(raw)

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        for key in sorted(self._data):
            if key.startswith(prefix):
                yield key, json.loads(self._data[key])

    def cursor(self) -> int:
        return self._cursor

    def epoch(self) -> int:
        return self._epoch

    def commit(
        self,
        expected: int,
        cursor: int,
        writes: Mapping[str, Any],
        deletes: AbstractSet[str],
        *,
        epoch: int,
    ) -> None:
        if self._cursor != expected or self._epoch != epoch:
            raise CursorConflict(
                f"cursor is {self._cursor} at epoch {self._epoch}, expected {expected} at epoch {epoch}"
            )
        encoded = {key: json.dumps(_copy(value)) for key, value in writes.items()}
        for key in deletes:
            self._data.pop(key, None)
        self._data.update(encoded)
        self._cursor = cursor
        self._epoch += 1

    def version(self) -> str | None:
        return self._version

    def position(self) -> tuple[int, int, str | None]:
        return self._epoch, self._cursor, self._version

    def reset(self, version: str | None) -> None:
        self._data.clear()
        self._cursor = 0
        self._epoch += 1
        self._version = version


class InMemoryVectorBackend:
    """Vectors in dicts; the searches score in Python and judge visibility
    with cortex_mem.access over the events and knowledge given here."""

    kind = "memory"

    def __init__(self, events: EventReader, knowledge: KnowledgeReader) -> None:
        """events and knowledge: the stores the searches judge visibility over."""
        self._events = events
        self._knowledge = knowledge
        self._vectors: dict[str, dict[str, list[float]]] = {
            "event": {},
            "knowledge": {},
        }
        self._groups: dict[str, tuple[str, Scope, str, int]] = {}
        # (seq, event_id) of each (scope, owner) cell, kept sorted on put_group so
        # grouping each new event does not sort every group again.
        self._cells: dict[tuple[Scope, str], list[tuple[int, str]]] = {}
        self._cache: dict[tuple[str, str], list[float]] = {}
        self._digests: dict[str, str] = {}
        self._meta: dict[str, Any] | None = None
        self._indexed = 0
        self._generation = 0

    def meta(self) -> dict[str, Any] | None:
        return None if self._meta is None else dict(self._meta)

    def reset(self, meta: dict[str, Any] | None) -> None:
        for table in self._vectors.values():
            table.clear()
        self._groups.clear()
        self._cells.clear()
        self._digests.clear()
        self._indexed = 0
        self._meta = None if meta is None else dict(meta)
        self._generation += 1

    def generation(self) -> int:
        return self._generation

    def indexed(self) -> int:
        return self._indexed

    def position(self) -> tuple[dict[str, Any] | None, int, int]:
        return self.meta(), self._generation, self._indexed

    def set_indexed(self, expected: int, seq: int, generation: int) -> None:
        if self._indexed != expected or self._generation != generation:
            raise CursorConflict(
                f"index at {self._indexed} (generation {self._generation}), "
                f"not {expected} (generation {generation})"
            )
        self._indexed = seq

    def put_vector(
        self, kind: str, id: str, vector: Sequence[float], digest: str | None = None
    ) -> None:
        # Assigning to an existing dict key keeps its original position.
        self._vectors[kind][id] = float32(vector)
        if kind == "knowledge":
            if digest is None:
                self._digests.pop(id, None)
            else:
                self._digests[id] = digest

    def knowledge_digests(self, ids: Collection[str] | None = None) -> dict[str, str]:
        if ids is None:
            return dict(self._digests)
        return {id: self._digests[id] for id in ids if id in self._digests}

    def vector(self, kind: str, id: str) -> list[float] | None:
        found = self._vectors[kind].get(id)
        return None if found is None else list(found)

    def vectors(self, kind: str, upto: int | None = None) -> Iterator[tuple[str, list[float]]]:
        table = self._vectors[kind]
        ids = sorted(table) if kind == "knowledge" else list(table)
        if upto is not None:
            if kind != "event":
                raise ValueError("upto bounds event vectors only")
            ids = [id for id in ids if id in self._groups and self._groups[id][3] <= upto]
        for id in ids:
            yield id, list(table[id])

    def delete_vectors(self, kind: str, ids: Collection[str]) -> None:
        for id in ids:
            self._vectors[kind].pop(id, None)
            if kind == "knowledge":
                self._digests.pop(id, None)

    def put_group(
        self, event_id: str, group_id: str, scope: Scope, owner: str, seq: int
    ) -> None:
        previous = self._groups.get(event_id)
        if previous is not None:
            self._cells[(previous[1], previous[2])].remove((previous[3], event_id))
        self._groups[event_id] = (group_id, scope, owner, seq)
        bisect.insort(self._cells.setdefault((scope, owner), []), (seq, event_id))

    def group_of(self, event_id: str) -> str | None:
        found = self._groups.get(event_id)
        return None if found is None else found[0]

    def members(self, group_id: str, upto: int | None = None) -> list[str]:
        rows = [
            (g[3], e)
            for e, g in self._groups.items()
            if g[0] == group_id and (upto is None or g[3] <= upto)
        ]
        return [e for _, e in sorted(rows)]

    def partition(self, scope: Scope, owner: str) -> list[str]:
        return [e for _, e in self._cells.get((scope, owner), [])]

    def cached(self, model: str, digest: str) -> list[float] | None:
        found = self._cache.get((model, digest))
        return None if found is None else list(found)

    def cache(self, model: str, digest: str, vector: Sequence[float]) -> None:
        self._cache[(model, digest)] = float32(vector)

    def _grouped(self, upto: int | None) -> list[str]:
        """Grouped event ids that have a vector, seq <= upto, by (seq, id)."""
        rows = sorted(
            (g[3], e)
            for e, g in self._groups.items()
            if e in self._vectors["event"] and (upto is None or g[3] <= upto)
        )
        return [e for _, e in rows]

    def nearest_member(
        self, scope: Scope, owner: str, vector: Sequence[float], upto: int
    ) -> tuple[str, float] | None:
        score_of = QueryScorer(float32(vector))
        table = self._vectors["event"]
        best: tuple[str, float] | None = None
        for seq, id in self._cells.get((scope, owner), []):  # (seq, id) order
            if seq > upto:
                break
            if id in table:
                score = score_of(table[id])
                if best is None or score > best[1]:  # strictly: a tie keeps the earlier
                    best = (id, score)
        return best

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
        if k <= 0 or (kinds is not None and not kinds):
            return []
        ids = [id for id in self._grouped(upto) if id != exclude]
        events = {e.id: e for e in self._events.get(ids)}
        table = self._vectors["event"]
        candidates = [
            (id, table[id])
            for id in ids
            if id in events
            and (kinds is None or events[id].kind in kinds)
            and event_visible(events[id], caller, scope)
        ]
        return top_by_cosine(float32(vector), candidates, k)

    def search_knowledge(
        self, vector: Sequence[float], caller: Caller, scope: Scope, k: int
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        table = self._vectors["knowledge"]
        ids = sorted(table)
        items = {item.id: item for item in self._knowledge.get(ids)}
        candidates = [
            (id, table[id]) for id in ids if id in items and knowledge_visible(items[id], caller, scope)
        ]
        return top_by_cosine(float32(vector), candidates, k)


class InMemoryDatabase:
    """Every store in memory: one per harness scenario run, or per test."""

    def __init__(self) -> None:
        self.events = InMemoryEventStore()
        self.knowledge = InMemoryKnowledgeBase()
        self.vectors = InMemoryVectorBackend(self.events, self.knowledge)
        self._states: dict[str, InMemoryStateBackend] = {}

    def state(self, engine: str) -> InMemoryStateBackend:
        return self._states.setdefault(engine, InMemoryStateBackend())

    def transaction(self) -> AbstractContextManager[object]:
        return nullcontext()
