"""A similarity port the test decides, instead of real embeddings."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from cortex_mem.access import event_visible, owner_for
from cortex_mem.contract import Caller, Event, Group, Kind, Knowledge, Scope
from cortex_mem.engines.ledger.text import jaccard


class FakeSimilarity:
    """Groups come from `groups` (event id -> group label). Like core, it only
    knows events up to log position `upto` and applies event visibility."""

    model = "fake"

    def __init__(
        self,
        events,
        groups: Mapping[str, str],
        scores: Mapping[frozenset[str], float] | None = None,
        default_score: float = 0.95,
        upto: int | None = None,
    ) -> None:
        self._groups = dict(groups)
        self._scores = dict(scores or {})
        self._default = default_score
        self._log: list[Event] = [
            e for pos, e in events.since(0) if upto is None or pos <= upto
        ]
        self._by_id = {e.id: e for e in self._log}

    def _members(self, label: str) -> list[Event]:
        return [e for e in self._log if self._groups.get(e.id) == label]

    def group(self, event_id: str) -> Group | None:
        label = self._groups.get(event_id)
        if label is None or event_id not in self._by_id:
            return None
        members = self._members(label)
        first = members[0]
        return Group(
            id=first.id,
            scope=first.scope,
            owner=owner_for(first),
            members=tuple(m.id for m in members),
        )

    def neighbors(
        self, event_id: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Event, float]]:
        label = self._groups.get(event_id)
        if label is None or event_id not in self._by_id:
            return []
        scored = [
            (e, self._scores.get(frozenset({event_id, e.id}), self._default))
            for e in self._members(label)
            if e.id != event_id and event_visible(e, caller, scope)
        ]
        scored.sort(key=lambda p: (-p[1], p[0].id))
        return scored[:k]

    def search(
        self,
        query: str,
        caller: Caller,
        scope: Scope,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        scored = [
            (e, jaccard(query, e.text))
            for e in self._log
            if e.id in self._groups
            and (kinds is None or e.kind in kinds)
            and event_visible(e, caller, scope)
        ]
        scored = [p for p in scored if p[1] > 0]
        scored.sort(key=lambda p: (-p[1], p[0].id))
        return scored[:k]

    def knowledge(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        return []
