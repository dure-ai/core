"""The similarity port the ledger engine reads: core's groups and neighbors."""

from __future__ import annotations

from cortex_mem.contract import EngineContext, Group, SimilarityReader

__all__ = ["Group", "SimilarityReader", "similar_of"]


def similar_of(ctx: EngineContext) -> SimilarityReader | None:
    """The context's similarity port, or None when embeddings are off."""
    return ctx.similar
