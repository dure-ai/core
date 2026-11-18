"""Test doubles that tell whether work ran inside a write transaction.

TxnSpy wraps a Database's transaction(): pass `spy.transaction` where the code
under test takes the write-transaction opener (Index(transaction=...),
Runtime(transaction)). `spy.open` is True while a write transaction is open,
whether it was opened through the spy or by the store itself (SQLite and
PostgreSQL count their own nesting depth).
"""

from contextlib import contextmanager

from cortex_mem.embed import EmbeddingError, HashEmbedder


class TxnSpy:
    def __init__(self, db):
        self.db = db
        self.depth = 0

    @contextmanager
    def transaction(self):
        with self.db.transaction():
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1

    @property
    def open(self) -> bool:
        return self.depth > 0 or getattr(self.db, "_depth", 0) > 0


class SpyEmbedder:
    """HashEmbedder that records every embed call with whether a write
    transaction was open at that moment. `fail_on` texts raise EmbeddingError;
    `during` (if set) is called with the texts before they are embedded, so a
    test can change the store between an embedding and its write."""

    def __init__(self, spy, inner=None, fail_on=(), during=None):
        self.spy = spy
        self.inner = inner or HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.fail_on = set(fail_on)
        self.during = during
        self.calls = []  # (texts, kind, transaction open)

    def embed(self, texts, kind):
        self.calls.append((list(texts), kind, self.spy.open))
        if self.during is not None:
            self.during(list(texts), kind)
        if any(t in self.fail_on for t in texts):
            raise EmbeddingError("embedding backend is down")
        return self.inner.embed(texts, kind)

    @property
    def texts(self):
        return [t for texts, _, _ in self.calls for t in texts]

    @property
    def batches(self):
        return [texts for texts, _, _ in self.calls]

    @property
    def inside(self):
        """The calls made while a write transaction was open (should be none)."""
        return [c for c in self.calls if c[2]]


class SpyBackend:
    """Delegates to a VectorBackend and records, for the methods named in
    `watch`, each call with whether a write transaction was open, plus a log of
    the vector and group writes."""

    def __init__(self, inner, spy, watch=("nearest_member", "group_of")):
        self._inner = inner
        self._spy = spy
        self._watch = set(watch)
        self.calls = []  # (method, transaction open)
        self.log = []  # ("put_vector", kind, id) | ("put_group", event_id) | ("delete_vectors", kind, ids)

    def __getattr__(self, name):
        target = getattr(self._inner, name)
        if name not in self._watch:
            return target

        def watched(*args, **kw):
            self.calls.append((name, self._spy.open))
            return target(*args, **kw)

        return watched

    def put_vector(self, kind, id, vector, digest=None):
        self.log.append(("put_vector", kind, id))
        self._inner.put_vector(kind, id, vector, digest)

    def put_group(self, event_id, *args, **kw):
        self.log.append(("put_group", event_id))
        self._inner.put_group(event_id, *args, **kw)

    def delete_vectors(self, kind, ids):
        self.log.append(("delete_vectors", kind, tuple(sorted(ids))))
        self._inner.delete_vectors(kind, ids)
