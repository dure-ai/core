"""Supersede engine (engine A v1): within a topic, hand off only the best-checked, newest item.

Builds on EvidenceEngine's ranks (self < checked < confirmed) and changes
handoff only: items on the same topic compete, the one with the higher rank
wins, the newer one breaks a tie, and the rest are not handed off. A newer
checked record pushes out an older one without anyone naming it, which is how
a rule goes stale when nobody says so. Only a checked or confirmed item hides
others; self records never hide anything, since who wrote last is not
evidence. Items with different conditions never compete; the caller's context
already filters those.

Verdicts work as in EvidenceEngine: refs name what was judged (contract 3-2),
and a verdict's text is not an item.

Nothing is demoted for losing: the loser may be right again under another
query or after the winner fails, and handoff is the only place it shows.

Topic grouping is a stand-in until embedding similarity (pgvector) replaces
`_same_topic`: the share of the shorter item's words found in the other. BM25
does not work here, because words both items share get a low IDF and only the
words that differ score. On the harness dev set, records within one scenario
had a median overlap of 0.35-0.50 and records from different scenarios 0.14
(p90 0.29), hence 0.3. Different topics in one repo will overlap more.
"""

from ..contract import Caller, EngineContext, Handoff, Knowledge, Scope
from ..store.memory import _words
from .evidence import _POOL, SELF, EvidenceEngine, _rank_key

_SAME_TOPIC = 0.3


class SupersedeEngine(EvidenceEngine):
    state_version = "1"

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        hits = [
            item for item, _ in ctx.knowledge.search(query, caller, scope, k=max(k, _POOL))
            if not any(key in caller.context and caller.context[key] != v for key, v in item.conditions.items())
        ]
        relevance = {item.id: i for i, item in enumerate(hits)}
        rank = {item.id: ctx.state.get(_rank_key(item.id)) or SELF for item in hits}
        at = {e.id: e.at for e in ctx.events.get([item.sources[0] for item in hits])}
        # Winner first within a topic: higher rank, then newer.
        candidates = sorted(hits, key=lambda item: (-rank[item.id], -at[item.sources[0]].timestamp()))
        kept: list[Knowledge] = []
        lost: set[str] = set()
        for item in candidates:
            if len(kept) == k:
                break
            if item.id in lost:
                continue
            kept.append(item)
            if rank[item.id] == SELF:
                continue  # recency alone is not evidence: an unchecked claim hides nothing
            lost |= {other.id for other in hits
                     if other.conditions == item.conditions and _same_topic(item, other)}
        kept.sort(key=lambda item: (-rank[item.id], relevance[item.id]))
        return [
            Handoff(id=item.id, content=item.body, sources=item.sources, status="knowledge",
                    conditions=item.conditions)
            for item in kept
        ]


def _same_topic(a: Knowledge, b: Knowledge) -> bool:
    x, y = set(_words(a.body)), set(_words(b.body))
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= _SAME_TOPIC
