"""Test doubles that tell whether work ran inside a write transaction.

TxnSpy wraps a Database's transaction(): pass `spy.transaction` where the code
under test takes the write-transaction opener (Index(transaction=...),
Runtime(transaction)). `spy.open` is True while a write transaction is open,
whether it was opened through the spy or by the store itself (SQLite and
PostgreSQL count their own nesting depth).
"""

from contextlib import contextmanager

from cortex_mem.embed import EmbeddingError, EmbeddingRejected, HashEmbedder


class TxnSpy:
    """Wraps a Database's transaction() and counts how deep the transactions
    opened through it are (see the module docstring)."""
    def __init__(self, db):
        self.db = db
        self.depth = 0

    @contextmanager
    def transaction(self):
        """db.transaction(), counted while it is open."""
        with self.db.transaction():
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1

    @property
    def open(self) -> bool:
        """True while a write transaction is open, through the spy or by the
        store itself."""
        return self.depth > 0 or getattr(self.db, "_depth", 0) > 0


class SpyEmbedder:
    """HashEmbedder that records every embed call with whether a write
    transaction was open at that moment. `fail_on` texts raise EmbeddingError;
    a `reject_on` text alone raises EmbeddingRejected (in a batch, EmbeddingError,
    as the Upstage embedder does); `during` (if set) is called with the texts
    before they are embedded, so a test can change the store between an
    embedding and its write."""

    def __init__(self, spy, inner=None, fail_on=(), during=None, reject_on=()):
        self.spy = spy
        self.inner = inner or HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.fail_on = set(fail_on)
        self.reject_on = set(reject_on)
        self.during = during
        self.calls = []  # (texts, kind, transaction open)

    def embed(self, texts, kind):
        self.calls.append((list(texts), kind, self.spy.open))
        if self.during is not None:
            self.during(list(texts), kind)
        if any(t in self.fail_on for t in texts):
            raise EmbeddingError("embedding backend is down")
        if any(t in self.reject_on for t in texts):
            if len(texts) == 1:
                raise EmbeddingRejected("the embedding API refused this text")
            raise EmbeddingError("the embedding API refused the request")
        return self.inner.embed(texts, kind)

    @property
    def texts(self):
        """Every text embedded, in call order."""
        return [t for texts, _, _ in self.calls for t in texts]

    @property
    def batches(self):
        """The texts of each embed call."""
        return [texts for texts, _, _ in self.calls]

    @property
    def inside(self):
        """The calls made while a write transaction was open (should be none)."""
        return [c for c in self.calls if c[2]]


class SpyEngine:
    """Promotes each event and records, for observe, tick and handoff, whether
    a write transaction was open while the engine ran. `on_tick` decisions are
    returned by tick."""

    def __init__(self, spy, on_tick=()):
        self.spy = spy
        self.on_tick = list(on_tick)
        self.calls = []  # (method, transaction open)

    def observe(self, event, ctx):
        from cortex_mem.contract import Knowledge, Promote

        self.calls.append(("observe", self.spy.open))
        ctx.state.put(event.id, True)
        item = Knowledge(id=f"k-{event.id}", kind="knowledge", title=event.id, body=event.text,
                         scope=event.scope, owner=event.user, sources=(event.id,))
        return [Promote(item)]

    def tick(self, now, ctx):
        self.calls.append(("tick", self.spy.open))
        return list(self.on_tick)

    def handoff(self, query, caller, scope, ctx, k=6):
        self.calls.append(("handoff", self.spy.open))
        return []

    @property
    def inside(self):
        """The engine calls made while a write transaction was open (should be none)."""
        return [c for c in self.calls if c[1]]


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
        """The inner backend's attribute; a watched method also records its calls."""
        target = getattr(self._inner, name)
        if name not in self._watch:
            return target

        def watched(*args, **kw):
            self.calls.append((name, self._spy.open))
            return target(*args, **kw)

        return watched

    def put_vector(self, kind, id, vector, digest=None):
        """Log the write, then delegate."""
        self.log.append(("put_vector", kind, id))
        self._inner.put_vector(kind, id, vector, digest)

    def put_group(self, event_id, *args, **kw):
        """Log the write, then delegate."""
        self.log.append(("put_group", event_id))
        self._inner.put_group(event_id, *args, **kw)

    def delete_vectors(self, kind, ids):
        """Log the delete, then delegate."""
        self.log.append(("delete_vectors", kind, tuple(sorted(ids))))
        self._inner.delete_vectors(kind, ids)
