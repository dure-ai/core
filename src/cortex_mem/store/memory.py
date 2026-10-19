"""In-memory EpisodeStore for tests and harness runs. Token-overlap scoring;
ties keep insertion order so results are deterministic."""

import re
from collections.abc import Collection

from ..contract import Event, Kind, QueryContext

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def visible(event: Event, ctx: QueryContext) -> bool:
    if event.user == ctx.user:
        return True
    return ctx.scope == "team" and event.scope == "team"


class InMemoryEpisodeStore:
    def __init__(self) -> None:
        self._events: dict[str, Event] = {}

    def append(self, event: Event) -> str:
        if event.id in self._events:
            raise ValueError(f"duplicate event id: {event.id}")
        self._events[event.id] = event
        return event.id

    def get(self, ids: Collection[str]) -> list[Event]:
        return [self._events[i] for i in ids if i in self._events]

    def search(
        self,
        query: str,
        ctx: QueryContext,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        q = _tokens(query)
        scored = []
        for e in self._events.values():
            if kinds is not None and e.kind not in kinds:
                continue
            if not visible(e, ctx):
                continue
            score = len(q & _tokens(e.text))
            if score:
                scored.append((e, float(score)))
        scored.sort(key=lambda pair: -pair[1])  # stable: ties keep insertion order
        return scored[:k]
