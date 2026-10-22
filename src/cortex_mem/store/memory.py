"""In-memory stores for tests and harness runs. Token-overlap scoring;
ties keep insertion order so results are deterministic."""

import json
import re
from collections.abc import Collection, Iterator, Mapping, Set
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ..access import event_visible, knowledge_visible
from ..contract import Caller, Event, Kind, Knowledge, Scope
from .base import CursorConflict

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


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
        q = _tokens(query)
        scored = []
        for e in self._log:
            if kinds is not None and e.kind not in kinds:
                continue
            if not event_visible(e, caller, scope):
                continue
            score = len(q & _tokens(e.text))
            if score:
                scored.append((e, float(score)))
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
        q = _tokens(query)
        scored = []
        for item in self.all():
            if not knowledge_visible(item, caller, scope):
                continue
            score = len(q & _tokens(f"{item.title} {item.body}"))
            if score:
                scored.append((item, float(score)))
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
