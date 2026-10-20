"""Optional helpers for engines. Nothing here is part of the port: an engine
that wants its own handoff strategy ignores this module."""

from .contract import Caller, EngineContext, Handoff, Knowledge, Scope


def knowledge_handoff(item: Knowledge) -> Handoff:
    return Handoff(
        id=item.id,
        content=item.body,
        sources=item.sources,
        status="knowledge",
        conditions=item.conditions,
    )


def handoff_from_knowledge(
    query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
) -> list[Handoff]:
    """Hand off the best-matching knowledge items, nothing that is not promoted."""
    return [knowledge_handoff(item) for item, _ in ctx.knowledge.search(query, caller, scope, k=k)]
