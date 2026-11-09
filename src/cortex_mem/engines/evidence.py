"""Evidence engine (engine A v0): hand off every record, the best-checked first.

Every record is promoted, as in PromoteAllEngine, but each item carries how
it was checked, and handoff puts better-checked items first, relevance
second. The agent tends to follow the first item, so ranking is where a
false note loses to a checked one.

- A record with an outside signal (test, review, human, doc) is checked.
- A passing outcome lifts the items it judges to checked.
- A human confirmation becomes an item of its own, ranked above checked, and
  replaces the items it refers to: a reviewer's account of reviewed notes is
  the newer, checked version of them, whether it agrees or corrects them.
- A failing outcome or confirmation removes the items it judges.
- An item whose conditions disagree with the caller's context is not handed off.

No model is used. Self records are not counted by how many people wrote them:
telling one claim from its contradiction needs more than word overlap.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime

from ..access import owner_for
from ..contract import (
    Caller,
    Decision,
    Demote,
    EngineContext,
    Event,
    Handoff,
    Knowledge,
    Promote,
    Scope,
)

SELF, CHECKED, CONFIRMED = 0, 1, 2
# Candidates read before reranking; more than k so a checked item found lower by relevance can move up.
_POOL = 50


def _rank_key(id: str) -> str:
    return f"rank:{id}"


class EvidenceEngine:
    state_version = "1"

    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        pass

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        if event.kind == "record":
            return [self._promote(event, SELF if event.signal in (None, "self") else CHECKED, ctx)]
        if event.kind not in ("outcome", "human_confirm"):
            return []
        judged = self._judged(event.refs, ctx)
        if event.outcome == "failure":
            return self._demote(judged, "a failure was reported", event, ctx)
        if event.outcome != "success":
            return []
        if event.kind == "outcome":
            for item in judged:
                ctx.state.put(_rank_key(item.id), max(CHECKED, ctx.state.get(_rank_key(item.id)) or SELF))
            return []
        return [self._promote(event, CONFIRMED, ctx), *self._demote(judged, "replaced by a review", event, ctx)]

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        return []

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        hits = [
            item for item, _ in ctx.knowledge.search(query, caller, scope, k=max(k, _POOL))
            if not any(key in caller.context and caller.context[key] != v for key, v in item.conditions.items())
        ]
        hits.sort(key=lambda item: -(ctx.state.get(_rank_key(item.id)) or SELF))  # stable: relevance next
        return [
            Handoff(id=item.id, content=item.body, sources=item.sources, status="knowledge",
                    conditions=item.conditions)
            for item in hits[:k]
        ]

    def _promote(self, event: Event, rank: int, ctx: EngineContext) -> Promote:
        id = f"k-{event.id}"
        ctx.state.put(_rank_key(id), rank)
        return Promote(
            Knowledge(
                id=id, kind="knowledge", title=event.text.splitlines()[0][:80] if event.text else event.id,
                body=event.text, scope=event.scope, owner=owner_for(event), sources=(event.id,),
                conditions=event.conditions,
            )
        )

    def _judged(self, refs: Sequence[str], ctx: EngineContext) -> list[Knowledge]:
        """Refs name an item or the event it came from."""
        return ctx.knowledge.get(list(dict.fromkeys(i for r in refs for i in (r, f"k-{r}"))))

    def _demote(self, items: list[Knowledge], reason: str, event: Event, ctx: EngineContext) -> list[Demote]:
        for item in items:
            ctx.state.delete(_rank_key(item.id))
        return [Demote(item.id, reason, (event.id,), since=event.at) for item in items]
