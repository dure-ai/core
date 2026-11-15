"""The embedding indexer: core embeds record events in append order, groups
similar records within one (scope, owner) cell, keeps knowledge vectors, and
hands engines a read-only SimilarityReader (design: core embedding §6).

Determinism: the same event order, the same vectors and the same threshold
give the same groups and the same search results. Ties go to append order
for events and id order for knowledge.
"""

from collections.abc import Collection, Iterator

from .access import event_visible, knowledge_visible, owner_for
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
from .embed import Embedder, EmbeddingError, EmbedKind, cosine, text_digest
from .store.base import EventStore, VectorBackend

_QUERY_CACHE_SIZE = 256


def _embedded(event: Event) -> bool:
    """Only records with words of their own are embedded (spec §4)."""
    return event.kind == "record" and not event.text_redacted and event.text != ""


def _passage(item: Knowledge) -> str:
    return f"{item.title}\n{item.body}"


class Index:
    """Writes vectors and groups through a VectorBackend. Every call that
    writes runs inside the runtime's transaction()."""

    def __init__(self, backend: VectorBackend, embedder: Embedder, group_threshold: float) -> None:
        self._backend = backend
        self._embedder = embedder
        self._threshold = group_threshold
        # Query vectors live only in this process, never in the backend.
        self._queries: dict[str, list[float]] = {}

    def want(self) -> dict:
        """The index version this Index builds."""
        return {"model": self._embedder.model, "group_threshold": self._threshold}

    def stale(self) -> bool:
        return self._backend.meta() != self.want()

    def rebuild(self, knowledge: KnowledgeReader) -> None:
        """Start over: drop vectors, groups and the cursor, then embed every
        knowledge item again (the cache makes that free for the same model)."""
        self._backend.reset(self.want())
        for item in knowledge.all():
            self.put_knowledge(item)

    def catch_up(self, events: EventStore, upto: int) -> None:
        """Index every event after backend.indexed() up to and including upto.
        An EmbeddingError leaves no trace of the event that failed."""
        for seq, event in events.since(self._backend.indexed()):
            if seq > upto:
                break
            if _embedded(event):
                # Embed and group first, write after: a failure writes nothing.
                vector = self._passage_vector(event.text)
                scope, owner = event.scope, owner_for(event)
                group_id = self._group_for(vector, scope, owner, event.id)
                self._backend.put_vector("event", event.id, vector)
                self._backend.put_group(event.id, group_id, scope, owner, seq)
            self._backend.set_indexed(seq)
            if seq >= upto:
                break

    def put_knowledge(self, item: Knowledge) -> None:
        self._backend.put_vector("knowledge", item.id, self._passage_vector(_passage(item)))

    def remove_knowledge(self, id: str) -> None:
        self._backend.delete_vectors("knowledge", [id])

    def reader(self, events: EventReader, knowledge: KnowledgeReader) -> SimilarityReader:
        return _Reader(self, events, knowledge)

    # ── internals ───────────────────────────────────────────────────────────

    def _passage_vector(self, text: str) -> list[float]:
        model, digest = self._embedder.model, text_digest(text)
        found = self._backend.cached(model, digest)
        if found is not None:
            return found
        vector = self._embed_one(text, "passage")
        self._backend.cache(model, digest, vector)
        return vector

    def _embed_one(self, text: str, kind: EmbedKind) -> list[float]:
        out = self._embedder.embed([text], kind)
        if len(out) != 1:
            raise EmbeddingError(f"embedder returned {len(out)} vectors for 1 text")
        return [float(x) for x in out[0]]

    def _group_for(self, vector: list[float], scope: Scope, owner: str, event_id: str) -> str:
        """Single link within the cell: join the group of the closest member if
        it reaches the threshold, else start a group named after this event.
        Members come in seq order and only a strictly higher score replaces the
        best, so a tie goes to the earlier member. Conditions are ignored."""
        best: str | None = None
        best_score = 0.0
        for member in self._backend.partition(scope, owner):
            other = self._backend.vector("event", member)
            if other is None:
                continue
            score = cosine(vector, other)
            if best is None or score > best_score:
                best, best_score = member, score
        if best is not None and best_score >= self._threshold:
            group_id = self._backend.group_of(best)
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

    def _event_vectors(self) -> Iterator[tuple[str, list[float]]]:
        return self._backend.vectors("event")

    def _knowledge_vectors(self) -> Iterator[tuple[str, list[float]]]:
        return self._backend.vectors("knowledge")

    def _group(self, event_id: str) -> tuple[str, list[str]] | None:
        group_id = self._backend.group_of(event_id)
        if group_id is None:
            return None
        return group_id, self._backend.members(group_id)

    def _vector(self, event_id: str) -> list[float] | None:
        return self._backend.vector("event", event_id)


def _rank[T](scored: list[tuple[T, float]], k: int) -> list[tuple[T, float]]:
    """Drop scores <= 0, best first; the sort is stable, so ties keep the
    order the candidates came in."""
    kept = [pair for pair in scored if pair[1] > 0]
    kept.sort(key=lambda pair: -pair[1])
    return kept[: max(k, 0)]


class _Reader:
    """SimilarityReader over an Index. Only read methods are public."""

    def __init__(self, index: Index, events: EventReader, knowledge: KnowledgeReader) -> None:
        self._index = index
        self._events = events
        self._knowledge = knowledge
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
        return self._rank_events(target, caller, scope, k, kinds, exclude=None)

    def neighbors(
        self, event_id: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Event, float]]:
        # Unknown or unseen start: even filtered scores would tell the caller
        # how close visible events are to text they may not read.
        start = self._events.get([event_id])
        if not start or not event_visible(start[0], caller, scope):
            return []
        target = self._index._vector(event_id)
        if target is None:
            return []
        return self._rank_events(target, caller, scope, k, None, exclude=event_id)

    def knowledge(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        if query.strip() == "":
            return []
        target = self._index._query_vector(query)
        stored = list(self._index._knowledge_vectors())  # id order
        items = {item.id: item for item in self._knowledge.get([id for id, _ in stored])}
        scored = [
            (items[id], cosine(target, vector))
            for id, vector in stored
            if id in items and knowledge_visible(items[id], caller, scope)
        ]
        return _rank(scored, k)

    def group(self, event_id: str) -> Group | None:
        found = self._index._group(event_id)
        if found is None:
            return None
        group_id, members = found
        first = self._events.get([members[0]]) if members else []
        if not first:
            return None
        return Group(group_id, first[0].scope, owner_for(first[0]), tuple(members))

    def _rank_events(
        self,
        target: list[float],
        caller: Caller,
        scope: Scope,
        k: int,
        kinds: Collection[Kind] | None,
        exclude: str | None,
    ) -> list[tuple[Event, float]]:
        stored = [(id, v) for id, v in self._index._event_vectors() if id != exclude]  # append order
        events = {e.id: e for e in self._events.get([id for id, _ in stored])}
        scored = [
            (events[id], cosine(target, vector))
            for id, vector in stored
            if id in events
            and (kinds is None or events[id].kind in kinds)
            and event_visible(events[id], caller, scope)
        ]
        return _rank(scored, k)
