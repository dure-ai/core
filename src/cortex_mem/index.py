"""The embedding indexer: core embeds record events in append order, groups
similar records within one (scope, owner) cell, keeps knowledge vectors, and
hands engines a read-only SimilarityReader (design: core embedding §6).

Indexing is a step of its own with its own position (indexed()), run before
an event's delivery and never inside a write transaction (design: delivery
outside the transaction §3.2-3.3). The embedder and the similarity reads run
with no transaction open; each write is a short transaction that first checks
the position and the index generation it read, so two processes indexing at
once write each event once.

Determinism: the same event order, the same vectors and the same threshold
give the same groups and the same search results. Ties go to append order
for events and id order for knowledge. The scores themselves are computed
by the VectorBackend (contract §5); this module embeds, caches, picks the
group and keeps the position.
"""

from collections.abc import Callable, Collection, Iterable
from contextlib import AbstractContextManager, nullcontext

from .access import event_visible, owner_for
from .contract import (
    Caller,
    Event,
    EventReader,
    Group,
    Kind,
    Knowledge,
    KnowledgeReader,
    Scope,
    SimilarityReader,
)
from .embed import Embedder, EmbeddingError, EmbedKind, text_digest
from .store.base import CursorConflict, EventStore, VectorBackend

_QUERY_CACHE_SIZE = 256
EMBED_BATCH = 64  # texts per embed call when filling the cache


class IndexConfigChanged(Exception):
    """The stored index was built for another config (model, threshold or
    store) than this Index's, or never built: advance refuses to write into
    it. The caller runs ensure_current and tries again."""


def _embedded(event: Event) -> bool:
    """Only records with words of their own are embedded (spec §4)."""
    return event.kind == "record" and not event.text_redacted and event.text != ""


def _passage(item: Knowledge) -> str:
    return f"{item.title}\n{item.body}"


class Index:
    """Writes vectors and groups through a VectorBackend.

    `transaction` opens one write transaction across the stores (a durable
    Database passes its db.transaction; the in-memory default needs none).
    advance, sync_knowledge and ensure_current must be called with no write
    transaction open: they embed first, with none open, and then write in
    short transactions of their own. The embedder never runs inside one."""

    def __init__(
        self,
        backend: VectorBackend,
        embedder: Embedder,
        group_threshold: float,
        transaction: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._threshold = group_threshold
        self._transaction = transaction
        # Query vectors live only in this process, never in the backend.
        self._queries: dict[str, list[float]] = {}

    def want(self) -> dict:
        """The index version this Index builds. It names the store, because
        stores may score in the last float32 digits differently: moving to
        another store rebuilds the index."""
        return {
            "model": self._embedder.model,
            "group_threshold": float(self._threshold),  # 1 and 1.0 name the same index
            "store": self._backend.kind,
        }

    def fingerprint(self, generation: int | None = None) -> dict:
        """want() plus the stored index's generation: what an engine's state
        built on this index depends on. `generation` is one the caller has
        read already; None reads it now. A rebuild, by any process and with
        any config, adds 1 to the generation, so it changes the fingerprint
        even when the config ends up the same."""
        if generation is None:
            generation = self._backend.generation()
        return {**self.want(), "generation": generation}

    def stale(self) -> bool:
        return self._backend.meta() != self.want()

    def generation(self) -> int:
        """The stored index's generation: one more after every rebuild."""
        return self._backend.generation()

    def indexed(self) -> int:
        """Position of the last event indexed (0 = none)."""
        return self._backend.indexed()

    def ensure_current(self, knowledge: KnowledgeReader) -> bool:
        """Rebuild the index when it is stale: drop vectors, groups and the
        position, add 1 to the generation, and store every knowledge item's
        vector again. True when this call rebuilt it; False when it was current
        (or another process rebuilt it first). Every passage is embedded
        before the one short write transaction (the cache makes that free for
        the same model), so an EmbeddingError changes nothing but the cache
        and the index stays stale."""
        if not self.stale():
            return False
        items = list(knowledge.all())
        vectors = self._knowledge_vectors(items)
        with self._transaction():
            if not self.stale():
                return False  # another process rebuilt it meanwhile
            self._backend.reset(self.want())
            self._put_knowledge(knowledge, vectors)
        return True

    def advance(self, events: EventStore, upto: int) -> None:
        """Index every event after indexed() up to and including upto.

        The passages missing from the cache are embedded first, in batches of
        at most EMBED_BATCH, with no transaction open (an EmbeddingError is
        raised before any event is written). Then each event in order: its
        group is found outside any transaction, and one short transaction
        moves the position from seq - 1 to seq, checked against the generation
        read at the start, before it writes the vector and the group. When
        another process moved the position first (CursorConflict), the
        position and generation are read again and the pass goes on from
        there, with the cache it filled. If a batch of several texts fails,
        they are embedded one at a time: the events before the first text
        that cannot be embedded are indexed, and then its EmbeddingError is
        raised (unless another process indexed that event meanwhile).

        Raises IndexConfigChanged, writing nothing more, when the stored index
        holds another config than this Index's, at the start or after a
        conflict: the caller runs ensure_current and tries again."""
        generation, position = self._read_position()
        limit = upto  # lowered to just before an event whose text cannot be embedded
        failure: tuple[int, EmbeddingError] | None = None  # (seq of that event, its error)
        unexplained = 0
        while limit > position:
            texts = [event.text for seq, event in self._pending(events, position, limit) if _embedded(event)]
            batch = self._cache_batches(texts)
            if batch is not None:
                chunk, exc = batch
                # Find the text that fails, unless the batch held only that one.
                found = (chunk[0], exc) if len(chunk) == 1 else self._first_failure(chunk)
                if found is not None:
                    seq = next(
                        seq for seq, event in self._pending(events, position, limit)
                        if _embedded(event) and event.text == found[0]
                    )
                    failure, limit = (seq, found[1]), seq - 1
                continue  # every text succeeded on its own: go on with them all
            try:
                for seq, event in self._pending(events, position, limit):
                    self._index_event(seq, event, generation)
                    position = seq
                break
            except CursorConflict:
                now = self._read_position()
                if now == (generation, position):
                    unexplained += 1
                    if unexplained > 1:
                        raise  # the position did not move, twice: nothing explains the conflict
                generation, position = now
                if failure is not None and position >= failure[0]:
                    failure, limit = None, upto  # another process indexed past the failure
        if failure is not None and self._backend.indexed() < failure[0]:
            raise failure[1]

    def sync_knowledge(self, knowledge: KnowledgeReader, touched: Collection[str] | None = None) -> None:
        """Make the knowledge vectors follow the knowledge base, for the ids in
        `touched` (None: every item, and every stored vector). An item whose
        stored digest differs from its passage's is embedded with no
        transaction open; an id no longer in the knowledge base loses its
        vector. One short transaction writes them, and writes a vector only
        if the item's passage is still the one that was embedded: if another
        process changed the item meanwhile, its own sync writes the right one.

        The generation and the config are checked before anything is read and
        again inside the write transaction: if the index is stale, or another
        process rebuilt it meanwhile, nothing is written and the call returns
        quietly. The rebuilder stores every knowledge vector of its snapshot,
        and a Runtime that rebuilds compares every item again afterwards, so
        nothing is lost; raising instead would only make the caller retry
        against the same rebuild."""
        generation = self._backend.generation()  # first: a rebuild after this read changes it
        if self.stale():
            return
        # Reads every stored digest (and, in full, every vector id): O(knowledge
        # base) per call. A per-id lookup on the port would avoid it.
        stored = self._backend.knowledge_digests()
        if touched is None:
            items = list(knowledge.all())
            # Ids of every stored vector: one without a digest is removed too.
            held = set(stored) | {id for id, _ in self._backend.vectors("knowledge")}
            gone = held - {item.id for item in items}
        else:
            items = knowledge.get(list(dict.fromkeys(touched)))
            gone = set(touched) - {item.id for item in items}
        changed = [item for item in items if stored.get(item.id) != text_digest(_passage(item))]
        if not changed and not gone:
            return
        vectors = self._knowledge_vectors(changed)
        with self._transaction():
            if self._backend.generation() != generation or self.stale():
                return  # rebuilt meanwhile, maybe for another model: not ours to write into
            still_gone = gone - {item.id for item in knowledge.get(list(gone))} if gone else set()
            if still_gone:
                self._backend.delete_vectors("knowledge", sorted(still_gone))
            self._put_knowledge(knowledge, vectors)

    def reader(
        self, events: EventReader, knowledge: KnowledgeReader, upto: int | None = None
    ) -> SimilarityReader:
        """A reader over the events at positions <= upto (all when None): the
        runtime passes the event being observed, or the engine's cursor, so an
        engine never reads events it has not been given, live or on replay.
        Knowledge is the knowledge base as it stands now."""
        return _Reader(self, events, knowledge, upto)

    # ── internals ───────────────────────────────────────────────────────────

    def _read_position(self) -> tuple[int, int]:
        """(generation, position) of the stored index, the generation read
        first: a rebuild after that read changes the generation, so a write
        checked against it conflicts. Raises IndexConfigChanged if the index
        holds another config than this Index's (a rebuild by another process
        with another model or threshold): its generation is not ours to write."""
        generation = self._backend.generation()
        if self.stale():
            raise IndexConfigChanged(f"the index was built for {self._backend.meta()}, not {self.want()}")
        return generation, self._backend.indexed()

    def _pending(self, events: EventStore, after: int, upto: int) -> Iterable[tuple[int, Event]]:
        for seq, event in events.since(after):
            if seq > upto:
                return
            yield seq, event

    def _index_event(self, seq: int, event: Event, generation: int) -> None:
        if not _embedded(event):
            with self._transaction():
                self._backend.set_indexed(seq - 1, seq, generation)
            return
        # Read outside the transaction: events up to seq - 1 are grouped for
        # good within this generation, which set_indexed checks below.
        vector = self._passage_vector(event.text)
        scope, owner = event.scope, owner_for(event)
        group_id = self._group_for(vector, scope, owner, event.id, seq)
        with self._transaction():
            self._backend.set_indexed(seq - 1, seq, generation)  # first: a conflict writes nothing
            self._backend.put_vector("event", event.id, vector)
            self._backend.put_group(event.id, group_id, scope, owner, seq)

    def _cache_passages(self, texts: Iterable[str]) -> None:
        """Embed the passages the cache lacks, at most EMBED_BATCH per embed
        call, with no transaction open, and cache each batch in its own short
        transaction. An EmbeddingError keeps the batches cached before it."""
        failed = self._cache_batches(texts)
        if failed is not None:
            raise failed[1]

    def _cache_batches(self, texts: Iterable[str]) -> tuple[list[str], EmbeddingError] | None:
        """_cache_passages, but a failed batch is returned with its error
        instead of raised: (its uncached texts, the error), else None."""
        model = self._embedder.model
        todo = [t for t in dict.fromkeys(texts) if self._backend.cached(model, text_digest(t)) is None]
        for i in range(0, len(todo), EMBED_BATCH):
            chunk = todo[i : i + EMBED_BATCH]
            try:
                vectors = self._embed(chunk, "passage")
            except EmbeddingError as exc:
                return chunk, exc
            with self._transaction():
                for text, vector in zip(chunk, vectors, strict=True):
                    self._backend.cache(model, text_digest(text), vector)
        return None

    def _first_failure(self, texts: list[str]) -> tuple[str, EmbeddingError] | None:
        """Embed and cache the uncached texts one at a time, in order; the first
        that fails, with its error (None if every one succeeds now)."""
        for text in dict.fromkeys(texts):
            try:
                self._cache_passages([text])
            except EmbeddingError as exc:
                return text, exc
        return None

    def _knowledge_vectors(self, items: list[Knowledge]) -> dict[str, tuple[str, list[float]]]:
        """id -> (passage digest, vector), embedding what the cache lacks."""
        self._cache_passages(_passage(item) for item in items)
        return {
            item.id: (text_digest(_passage(item)), self._passage_vector(_passage(item))) for item in items
        }

    def _put_knowledge(self, knowledge: KnowledgeReader, vectors: dict[str, tuple[str, list[float]]]) -> None:
        """Inside a write transaction: store each vector whose item still has
        the passage it was made from."""
        current = {item.id: item for item in knowledge.get(list(vectors))} if vectors else {}
        for id, (digest, vector) in vectors.items():
            item = current.get(id)
            if item is not None and text_digest(_passage(item)) == digest:
                self._backend.put_vector("knowledge", id, vector, digest)

    def _passage_vector(self, text: str) -> list[float]:
        """From the cache, else embedded and cached. Never call it inside a
        write transaction: on a miss it embeds."""
        model, digest = self._embedder.model, text_digest(text)
        found = self._backend.cached(model, digest)
        if found is not None:
            return found
        vector = self._embed_one(text, "passage")
        self._backend.cache(model, digest, vector)
        return vector

    def _embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        out = self._embedder.embed(texts, kind)
        if len(out) != len(texts):
            raise EmbeddingError(f"embedder returned {len(out)} vectors for {len(texts)} texts")
        return [[float(x) for x in vector] for vector in out]

    def _embed_one(self, text: str, kind: EmbedKind) -> list[float]:
        return self._embed([text], kind)[0]

    def _group_for(self, vector: list[float], scope: Scope, owner: str, event_id: str, seq: int) -> str:
        """Single link within the cell: join the group of the closest earlier
        member if it reaches the threshold, else start a group named after this
        event. The backend breaks a tie toward the earlier member. Conditions
        are ignored."""
        found = self._backend.nearest_member(scope, owner, vector, seq - 1)
        if found is not None and found[1] >= self._threshold:
            group_id = self._backend.group_of(found[0])
            if group_id is not None:
                return group_id
        return event_id

    def _query_vector(self, query: str) -> list[float]:
        """Embed a query once per process; an LRU of _QUERY_CACHE_SIZE entries."""
        found = self._queries.pop(query, None)
        if found is None:
            found = self._embed_one(query, "query")
        self._queries[query] = found  # most recently used goes last
        while len(self._queries) > _QUERY_CACHE_SIZE:
            del self._queries[next(iter(self._queries))]
        return found

    def _group(self, event_id: str, upto: int | None) -> tuple[str, list[str]] | None:
        group_id = self._backend.group_of(event_id)
        if group_id is None:
            return None
        members = self._backend.members(group_id, upto)
        if event_id not in members:
            return None  # the event itself is past upto
        return group_id, members


class _Reader:
    """SimilarityReader over an Index. Only read methods are public."""

    def __init__(
        self, index: Index, events: EventReader, knowledge: KnowledgeReader, upto: int | None
    ) -> None:
        self._index = index
        self._events = events
        self._knowledge = knowledge
        self._upto = upto
        self.model: str = index.want()["model"]

    def search(
        self,
        query: str,
        caller: Caller,
        scope: Scope,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        if query.strip() == "":
            return []
        target = self._index._query_vector(query)
        hits = self._index._backend.search_events(target, caller, scope, k, kinds, self._upto)
        return self._with_events(hits)

    def neighbors(
        self, event_id: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Event, float]]:
        # Unknown or unseen start: even filtered scores would tell the caller
        # how close visible events are to text they may not read.
        start = self._events.get([event_id])
        if not start or not event_visible(start[0], caller, scope):
            return []
        if self._index._group(event_id, self._upto) is None:
            return []  # not embedded, or past upto: every embedded event has a group
        target = self._index._backend.vector("event", event_id)
        if target is None:
            return []
        hits = self._index._backend.search_events(
            target, caller, scope, k, None, self._upto, exclude=event_id
        )
        return self._with_events(hits)

    def knowledge(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        if query.strip() == "":
            return []
        target = self._index._query_vector(query)
        hits = self._index._backend.search_knowledge(target, caller, scope, k)
        items = {item.id: item for item in self._knowledge.get([id for id, _ in hits])}
        return [(items[id], score) for id, score in hits if id in items]

    def group(self, event_id: str) -> Group | None:
        found = self._index._group(event_id, self._upto)
        if found is None:
            return None
        group_id, members = found
        first = self._events.get([members[0]]) if members else []
        if not first:
            return None
        return Group(group_id, first[0].scope, owner_for(first[0]), tuple(members))

    def _with_events(self, hits: list[tuple[str, float]]) -> list[tuple[Event, float]]:
        events = {e.id: e for e in self._events.get([id for id, _ in hits])}
        return [(events[id], score) for id, score in hits if id in events]
