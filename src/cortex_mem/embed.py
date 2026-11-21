"""Embedding port with cosine similarity, a hash embedder and an Upstage client."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import sys
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable, Sequence
from typing import Literal, Protocol

from cortex_mem.store.text import bigrams, words

EmbedKind = Literal["passage", "query"]
"""Whether a text is stored (passage) or searched with (query)."""


class EmbeddingError(Exception):
    """Raised when an embedding backend fails. The text may well embed later:
    the index retries it and holds back what depends on it."""


class EmbeddingRejected(EmbeddingError):
    """The backend works but refuses this one text (too long, for example),
    so retrying cannot help: the index leaves that text without a vector.
    Raised only for a single text, never for a batch."""


class Embedder(Protocol):
    """Port for anything that turns texts into vectors."""

    model: str

    @property
    def dim(self) -> int:
        """Vector length; read-only, since an embedder may learn it from its first response."""
        ...

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """One vector per text, in order, each `dim` long. kind says whether the
        texts are stored (passage) or searched with (query); a model may embed
        the two differently. Raises EmbeddingError when the backend fails."""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either is a zero vector.
    Raises ValueError when the lengths differ (vectors of two models).

    One square root of the product of the squared norms: for identical
    vectors that is the squared norm itself, so the result is exactly 1.0.

    The sums are math.sumprod, which sums the exact products in extended
    precision. The stores pass float32 values, whose products are exact in
    a double, so this is bit for bit the correctly rounded math.fsum of the
    products, about ten times faster."""
    return _cosine(a, math.sumprod(a, a), b)


def _cosine(a: Sequence[float], na2: float, b: Sequence[float]) -> float:
    """cosine(a, b) with a's squared norm given, for scoring one query
    against many vectors."""
    if len(a) != len(b):
        raise ValueError(f"cosine of vectors with different lengths: {len(a)} and {len(b)}")
    nb2 = math.sumprod(b, b)
    if na2 == 0.0 or nb2 == 0.0:
        return 0.0
    norms = math.sqrt(na2 * nb2)
    if norms < sys.float_info.min or math.isinf(norms):
        # The squares or their product left float range (overflowed or went
        # subnormal), so scale each vector by its largest component first.
        ma = max(abs(x) for x in a)
        mb = max(abs(y) for y in b)
        sa = [x / ma for x in a]
        sb = [y / mb for y in b]
        return math.sumprod(sa, sb) / (math.sqrt(math.sumprod(sa, sa)) * math.sqrt(math.sumprod(sb, sb)))
    return math.sumprod(a, b) / norms


class QueryScorer:
    """Scores one query against many vectors, its norm computed once."""

    def __init__(self, query: Sequence[float]) -> None:
        self.query = query
        self._na2 = math.sumprod(query, query)

    def __call__(self, vector: Sequence[float]) -> float:
        """cosine(query, vector)."""
        return _cosine(self.query, self._na2, vector)


def float32(vector: Iterable[float]) -> list[float]:
    """The vector rounded to float32, as every store keeps vectors (contract
    §5), back as Python floats. Raises ValueError on NaN, an infinity, or a
    value beyond float32 range (which would round to an infinity): a NaN
    compares false with everything and would silently break the rankings."""
    values = array("f", vector).tolist()
    if not all(math.isfinite(x) for x in values):
        raise ValueError("a vector must have finite values within float32 range")
    return values


def top_by_cosine[T](
    query: Sequence[float], candidates: Iterable[tuple[T, Sequence[float]]], k: int
) -> list[tuple[T, float]]:
    """The k candidates closest to query with a score > 0, best first. The sort
    is stable, so ties keep the order the candidates came in. The stores that
    score in Python share this, so they rank the same."""
    score = QueryScorer(query)
    scored = [(key, score(vector)) for key, vector in candidates]
    kept = [pair for pair in scored if pair[1] > 0]
    kept.sort(key=lambda pair: -pair[1])
    return kept[: max(k, 0)]


def text_digest(text: str) -> str:
    """Hex sha256 digest of the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashEmbedder:
    """Deterministic offline embedder using signed hashed token counts."""

    def __init__(self, dim: int = 256) -> None:
        """dim: vector length. The model name is `hash-<dim>`, so another dim is
        another index version and the index is rebuilt."""
        self.dim = dim
        self.model = f"hash-{dim}"

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Embed each text; kind has no effect."""
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        """Each word and bigram adds +1 or -1 (the sign from the same sha256) at
        the slot its hash picks; the result is L2-normalized, or all zeros for
        a text with no tokens."""
        vec = [0.0] * self.dim
        for token in words(text) + bigrams(text):
            n = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            vec[n % self.dim] += 1.0 if (n // self.dim) % 2 == 0 else -1.0
        norm = math.sqrt(math.fsum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


class UpstageEmbedder:
    """Embedder backed by the Upstage embeddings HTTP API."""

    # Statuses that may mean "this input", not "this service" or "this key".
    REJECTING = frozenset({400, 413, 422})
    # Embedded to tell a refused text from a model or request the API refuses outright.
    PROBE = "ping"

    def __init__(
        self,
        api_key: str,
        passage_model: str,
        query_model: str,
        base_url: str = "https://api.upstage.ai/v1",
        timeout: float = 30.0,
        batch: int = 64,
    ) -> None:
        """passage_model embeds stored texts and query_model search queries;
        `model`, the name the index version records, is the passage model.
        Texts go `batch` per request (at least 1), each request waiting at most
        `timeout` seconds. Raises ValueError, without quoting the key, for an
        api_key that is empty, has surrounding whitespace, or holds a character
        outside printable ASCII."""
        # The key goes into a header: a control or non-ASCII character would make
        # http.client raise an error that quotes the header, key included.
        # The message never quotes the key.
        if not api_key or api_key != api_key.strip() or not all(" " <= c <= "~" for c in api_key):
            raise ValueError("the API key must be non-empty printable ASCII, without control characters")
        self._api_key = api_key
        self.passage_model = passage_model
        self.query_model = query_model
        self.model = passage_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch = max(1, batch)
        self._dim: int | None = None
        self._working: set[str] = set()  # models that have answered in this process

    @property
    def dim(self) -> int:
        """Vector length learned from the first successful response."""
        if self._dim is None:
            raise ValueError("dim is unknown until the first successful embed call")
        return self._dim

    def __repr__(self) -> str:
        """Names the models only, never the API key."""
        return f"UpstageEmbedder(passage_model={self.passage_model!r}, query_model={self.query_model!r})"

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Embed texts in batches with the model matching kind.

        A single text answered with a status in REJECTING raises
        EmbeddingRejected once the model is known to work: it has answered
        before in this process, or it answers the PROBE text now. Otherwise,
        and for a batch of several texts, the error is an EmbeddingError."""
        model = self.passage_model if kind == "passage" else self.query_model
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i : i + self.batch])
            try:
                out.extend(self._post(model, chunk))
            except _Refused as exc:
                if len(chunk) == 1 and self._works(model):
                    raise EmbeddingRejected(f"the embedding API refused this text with HTTP {exc.code}") from None
                raise EmbeddingError(f"embedding request failed with HTTP {exc.code}") from None
        return out

    def _works(self, model: str) -> bool:
        """Whether `model` has embedded something in this process, or embeds PROBE now."""
        if model in self._working:
            return True
        try:
            self._post(model, [self.PROBE])
        except EmbeddingError:
            return False
        return True

    def _post(self, model: str, chunk: list[str]) -> list[list[float]]:
        """One POST /embeddings for the chunk; its vectors in input order, by the
        response's `index` field. Learns `dim` from the first non-empty response.

        Every failure is an EmbeddingError whose message never holds the key:
        an HTTP error, a timeout, a network error, a request that cannot be
        built or sent, or a response that is not valid embeddings JSON
        (missing, repeated or extra indexes, empty vectors, or lengths that
        differ from each other or from `dim`)."""
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        unsendable: str | None = None
        raw = b""  # set by the request; every path that leaves it unset raises below
        try:
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code in self.REJECTING:
                raise _Refused(exc.code) from None
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
        except ValueError as exc:  # UnicodeEncodeError included
            # http.client quotes the offending header, key included: keep only the type.
            unsendable = type(exc).__name__
        if unsendable is not None:
            # Raised outside the except block, so the original error (which may
            # quote the key) is not even kept as __context__.
            raise EmbeddingError(f"embedding request could not be built or sent ({unsendable})")
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
        self._working.add(model)
        return vectors


class _Refused(EmbeddingError):
    """Inside UpstageEmbedder: an HTTP status in REJECTING, before embed decides
    whether the text or the request is at fault."""

    def __init__(self, code: int) -> None:
        super().__init__(f"embedding request failed with HTTP {code}")
        self.code = code
