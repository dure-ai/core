"""Baseline engines from the harness draft: no memory and promote-everything."""

from collections.abc import Mapping, Sequence
from datetime import datetime

from ..access import owner_for
from ..contract import Caller, Decision, EngineContext, Event, Handoff, Knowledge, Promote, Scope
from ..kit import handoff_from_knowledge


class NoMemoryEngine:
    """Never promotes or hands anything off. The floor for repeated-mistake reduction."""

    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        pass

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        return []

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        return []

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        return []


class PromoteAllEngine:
    """Promotes every record immediately, shared exactly as widely as it was
    recorded. The "store everything" baseline."""

    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        pass

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        if event.kind != "record":
            return []
        item = Knowledge(
            id=f"k-{event.id}",
            kind="knowledge",
            title=event.text.splitlines()[0][:80] if event.text else event.id,
            body=event.text,
            scope=event.scope,
            owner=owner_for(event),
            sources=(event.id,),
            conditions=event.conditions,
        )
        return [Promote(item)]

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        return []

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        return handoff_from_knowledge(query, caller, scope, ctx, k)
