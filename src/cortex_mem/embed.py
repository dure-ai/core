"""Embedding port with cosine similarity, a hash embedder and an Upstage client."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Literal, Protocol

from cortex_mem.store.sqlite import _bigrams, _words

EmbedKind = Literal["passage", "query"]
"""Whether a text is stored (passage) or searched with (query)."""


class EmbeddingError(Exception):
    """Raised when an embedding backend fails."""


class Embedder(Protocol):
    """Port for anything that turns texts into vectors."""

    model: str
    dim: int

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either is a zero vector."""
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b)) / (na * nb)


def text_digest(text: str) -> str:
    """Hex sha256 digest of the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashEmbedder:
    """Deterministic offline embedder using signed hashed token counts."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model = f"hash-{dim}"

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Embed each text; kind has no effect."""
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _words(text) + _bigrams(text):
            n = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            vec[n % self.dim] += 1.0 if (n // self.dim) % 2 == 0 else -1.0
        norm = math.sqrt(math.fsum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]
