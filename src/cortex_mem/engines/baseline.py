"""Baseline engines from the harness draft: no memory and promote-everything."""

from datetime import datetime

from ..contract import EpisodeReader, Event, Handoff, QueryContext


def _as_handoff(event: Event) -> Handoff:
    return Handoff(
        id=event.id,
        content=event.text,
        sources=(event.id,),
        status="knowledge",
        conditions=event.conditions,
    )


class NoMemoryEngine:
    """Never hands anything off. The floor for repeated-mistake reduction."""

    def __init__(self, reader: EpisodeReader) -> None:
        self.reader = reader

    def observe(self, event: Event) -> None:
        pass

    def handoff(self, query: str, ctx: QueryContext, k: int = 6) -> list[Handoff]:
        return []

    def snapshot(self, at: datetime) -> list[Handoff]:
        return []


class PromoteAllEngine:
    """Treats every record as knowledge immediately. The "store everything" baseline."""

    def __init__(self, reader: EpisodeReader) -> None:
        self.reader = reader
        self._records: list[Event] = []

    def observe(self, event: Event) -> None:
        if event.kind == "record":
            self._records.append(event)

    def handoff(self, query: str, ctx: QueryContext, k: int = 6) -> list[Handoff]:
        hits = self.reader.search(query, ctx, k=k, kinds={"record"})
        return [_as_handoff(e) for e, _ in hits]

    def snapshot(self, at: datetime) -> list[Handoff]:
        return [_as_handoff(e) for e in self._records if e.at <= at]
