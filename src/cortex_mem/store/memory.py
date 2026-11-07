"""In-memory stores for tests and harness runs. BM25 over the visible documents,
with the SQLite store's tokens; ties keep append order for events and id order
for knowledge, so results are deterministic."""

import json
import math
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ..access import event_visible, knowledge_visible
from ..contract import Caller, Event, Kind, Knowledge, Scope
from .base import CursorConflict
from .sqlite import _bigrams, _words

# The harness's all_store baseline uses these, so a score gap between it and an
# engine on this store comes from the engine rather than from the ranking formula.
_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    return _words(text) + _bigrams(text)


def _bm25(query: str, texts: Sequence[str]) -> list[float]:
    """Score each text against the query; `texts` is the whole corpus for IDF."""
    docs = [Counter(_tokens(text)) for text in texts]
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

    def put(self, item: Knowledge) -> None:
        self._items[item.id] = item

    def remove(self, id: str) -> None:
        self._items.pop(id, None)

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

    def commit(
        self, expected: int, cursor: int, writes: Mapping[str, Any], deletes: Set[str]
    ) -> None:
        if self._cursor != expected:
            raise CursorConflict(f"cursor is {self._cursor}, expected {expected}")
        encoded = {key: json.dumps(_copy(value)) for key, value in writes.items()}
        for key in deletes:
            self._data.pop(key, None)
        self._data.update(encoded)
        self._cursor = cursor

    def version(self) -> str | None:
        return self._version

    def reset(self, version: str) -> None:
        self._data.clear()
        self._cursor = 0
        self._version = version


class InMemoryDatabase:
    """Every store in memory: one per harness scenario run, or per test."""

    def __init__(self) -> None:
        self.events = InMemoryEventStore()
        self.knowledge = InMemoryKnowledgeBase()
        self._states: dict[str, InMemoryStateBackend] = {}

    def state(self, engine: str) -> InMemoryStateBackend:
        return self._states.setdefault(engine, InMemoryStateBackend())

    def transaction(self) -> AbstractContextManager[object]:
        return nullcontext()
