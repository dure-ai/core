"""Embedding port with cosine similarity, a hash embedder and an Upstage client."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import urllib.error
import urllib.request
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


class UpstageEmbedder:
    """Embedder backed by the Upstage embeddings HTTP API."""

    def __init__(
        self,
        api_key: str,
        passage_model: str,
        query_model: str,
        base_url: str = "https://api.upstage.ai/v1",
        timeout: float = 30.0,
        batch: int = 64,
    ) -> None:
        self._api_key = api_key
        self.passage_model = passage_model
        self.query_model = query_model
        self.model = passage_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch = max(1, batch)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Vector length learned from the first successful response."""
        if self._dim is None:
            raise ValueError("dim is unknown until the first successful embed call")
        return self._dim

    def __repr__(self) -> str:
        return f"UpstageEmbedder(passage_model={self.passage_model!r}, query_model={self.query_model!r})"

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Embed texts in batches with the model matching kind."""
        model = self.passage_model if kind == "passage" else self.query_model
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._post(model, list(texts[i : i + self.batch])))
        return out

    def _post(self, model: str, chunk: list[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            raise EmbeddingError(f"embedding request failed with HTTP {exc.code}") from None
        except TimeoutError:
            raise EmbeddingError(f"embedding request timed out after {self.timeout}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise EmbeddingError(f"embedding request timed out after {self.timeout}s") from None
            raise EmbeddingError(f"embedding request failed: {exc.reason}") from None
        except http.client.HTTPException as exc:
            raise EmbeddingError(f"embedding response was malformed: {type(exc).__name__}") from None
        except OSError as exc:
            raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from None
        try:
            items = json.loads(raw)["data"]
            by_index = {item["index"]: item["embedding"] for item in items}
            if len(items) != len(by_index):
                raise EmbeddingError("embedding response repeats an index")
            if set(by_index) != set(range(len(chunk))):
                raise EmbeddingError(f"expected indexes 0..{len(chunk) - 1} in the embedding response")
            vectors = [[float(x) for x in by_index[i]] for i in range(len(chunk))]
        except EmbeddingError:
            raise
        except (ValueError, KeyError, TypeError):
            raise EmbeddingError("embedding response is not valid embeddings JSON") from None
        width = self._dim if self._dim is not None else len(vectors[0]) if vectors else 0
        if vectors and (width == 0 or any(len(v) != width for v in vectors)):
            raise EmbeddingError("embedding response has empty or inconsistent vector lengths")
        if self._dim is None and vectors:
            self._dim = width
        return vectors
