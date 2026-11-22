"""Supersede engine (engine A v1): within a topic, hand off only the best-checked, newest item.

Builds on EvidenceEngine's ranks (self < checked < confirmed) and changes
handoff only: a checked or confirmed item hides, on its topic, every self
item and every older item of its rank or lower. A newer checked record pushes
out an older one without anyone naming it, which is how a rule goes stale when
nobody says so. Self records never hide anything, since who wrote last is not
evidence, and an older item never hides a newer one, even when it is better
checked: a confirmed rule can go stale too. Items with different conditions
never compete; the caller's context already filters those. What is left is
ranked and cut to k as in EvidenceEngine.

Verdicts work as in EvidenceEngine: refs name what was judged (contract 3-2),
and a verdict's text is not an item.

Nothing is demoted for losing: the loser may be right again under another
query or after the winner fails, and handoff is the only place it shows.

With embeddings enabled, records compete only when the Core index puts their
source events in the same group. A source without a group does not compete.
Without embeddings, the topic test is the share of the shorter item's words
found in the other. On the harness dev set, records within one scenario had
a median overlap of 0.35-0.50 and those from different scenarios 0.14
(p90 0.29), hence 0.3. Different topics in one repo can overlap more.
"""

import re

from ..contract import Caller, EngineContext, Handoff, Knowledge, Scope
from .evidence import _POOL, SELF, EvidenceEngine, _rank_key

_SAME_TOPIC = 0.3
_WORD = re.compile(r"\w+")


class SupersedeEngine(EvidenceEngine):
    state_version = "1"

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        hits = [
            item for item, _ in ctx.knowledge.search(query, caller, scope, k=max(k, _POOL))
            if not any(key in caller.context and caller.context[key] != v for key, v in item.conditions.items())
        ]
        rank = {item.id: ctx.state.get(_rank_key(item.id)) or SELF for item in hits}
        at = {e.id: e.at for e in ctx.events.get([item.sources[0] for item in hits])}
        similar = ctx.similar
        groups = {
            item.id: group.id for item in hits
            if (group := similar.group(item.sources[0])) is not None
        } if similar is not None else {}
        words = {item.id: set(_WORD.findall(item.body.lower())) for item in hits} if similar is None else {}

        def hides(winner: Knowledge, other: Knowledge) -> bool:
            if rank[winner.id] == SELF or other.id == winner.id or other.conditions != winner.conditions:
                return False
            if rank[other.id] != SELF and not (
                rank[other.id] <= rank[winner.id] and at[other.sources[0]] < at[winner.sources[0]]
            ):
                return False
            if similar is not None:
                return winner.id in groups and groups[winner.id] == groups.get(other.id)
            return _same_topic(words[winner.id], words[other.id])

        kept = [item for item in hits if not any(hides(winner, item) for winner in hits)]
        kept.sort(key=lambda item: -rank[item.id])  # stable: relevance next
        return [
            Handoff(id=item.id, content=item.body, sources=item.sources, status="knowledge",
                    conditions=item.conditions)
            for item in kept[:k]
        ]


def _same_topic(x: set[str], y: set[str]) -> bool:
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= _SAME_TOPIC
