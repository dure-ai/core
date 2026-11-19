from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from cortex_mem.contract import Caller, Event
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.matcher import (
    GroupMatcher,
    LexicalMatcher,
    group_of,
    match_claim,
)
from cortex_mem.engines.ledger.model import Claim, Evidence, Memory
from cortex_mem.engines.ledger.similar import similar_of
from cortex_mem.engines.ledger.store import LedgerStore
from cortex_mem.engines.ledger.text import signature
from cortex_mem.store.memory import InMemoryDatabase

from .fakes import FakeSimilarity
from .test_store import FakeState

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 19, 10, 0, tzinfo=KST)
CFG = LedgerConfig.from_mapping(None)
LONG = "the deploy script must run migrations before restarting the api server every time"
LONG2 = "the deploy script must run migrations before restarting the api server each time"


@dataclass
class Ctx:
    events: Any
    knowledge: Any
    state: Any
    similar: Any = None


def rec(eid, text, *, user="u1", team="t1", scope="team", conditions=None):
    return Event(id=eid, kind="record", at=T0, user=user, team=team, session="s",
                 scope=scope, text=text, conditions=conditions or {})


class World:
    def __init__(self, cfg=CFG):
        self.db = InMemoryDatabase()
        self.state = FakeState()
        self.store = LedgerStore(self.state)
        self.cfg = cfg

    def add(self, event, *, claim_of=None, status="candidate", group=None):
        """Append an event; with claim_of=None it opens a new claim, else joins it."""
        self.db.events.append(event)
        from cortex_mem.access import owner_for

        cid = claim_of or f"c-{event.id}"
        if claim_of is None:
            self.store.put_claim(Claim(
                id=cid, scope=event.scope, owner=owner_for(event),
                conditions=dict(event.conditions), canonical=event.id, status=status,
                first_at=T0, last_support_at=T0, invalid_since=None,
                promoted=False, use_success=0, use_failure=0))
            if group:
                self.store.add_to_group(group, cid)
            self.store.index_signature(event.scope, owner_for(event),
                                       signature(event.text), cid)
        else:
            self.store.put_evidence(Evidence(
                claim_id=cid, event_id=event.id, stance="support",
                channel="observation", key=event.user, user=event.user, at=T0,
                derived=None))
        self.store.put_memory(Memory(id=event.id, claim_id=cid, user=event.user,
                                     team=event.team, session=event.session,
                                     at=T0, group=group))
        return event

    def ctx(self, groups=None, **kw):
        sim = None if groups is None else FakeSimilarity(self.db.events, groups, **kw)
        return Ctx(self.db.events, self.db.knowledge, self.state, sim)


def gm_match(w, event, groups, cfg=CFG, **kw):
    w.db.events.append(event)
    ctx = w.ctx(groups, **kw)
    return GroupMatcher(cfg).match(event, ctx, w.store)


def test_group_other_author_other_sentence():
    w = World()
    w.add(rec("e1", "use pnpm for installs"), group="e1")
    e2 = rec("e2", "pnpm is the package manager here", user="u2")
    m = gm_match(w, e2, {"e1": "g", "e2": "g"})
    assert m is not None and m.claim_id == "c-e1" and m.copied_from is None


def test_group_different_conditions():
    w = World()
    w.add(rec("e1", "use pnpm", conditions={"os": "mac"}), group="e1")
    e2 = rec("e2", "use pnpm", user="u2", conditions={"os": "linux"})
    assert gm_match(w, e2, {"e1": "g", "e2": "g"}) is None


def test_group_deprecated_claim_skipped():
    w = World()
    w.add(rec("e1", "use pnpm"), group="e1", status="deprecated")
    e2 = rec("e2", "pnpm please", user="u2")
    assert gm_match(w, e2, {"e1": "g", "e2": "g"}) is None


def test_group_threshold():
    cfg = LedgerConfig.from_mapping({"similar.same_threshold": "0.9"})
    for score, found in ((0.8, False), (0.95, True)):
        w = World(cfg)
        w.add(rec("e1", "use pnpm"), group="e1")
        e2 = rec("e2", "pnpm please", user="u2")
        m = gm_match(w, e2, {"e1": "g", "e2": "g"}, cfg=cfg,
                     scores={frozenset({"e1", "e2"}): score})
        assert (m is not None) is found
        if m:
            assert m.claim_id == "c-e1"


def test_group_prefers_higher_neighbor_score():
    w = World()
    w.add(rec("e1", "alpha topic"), group="e1")
    w.add(rec("e2", "beta topic", user="u2"), group="e1")
    e3 = rec("e3", "gamma topic", user="u3")
    m = gm_match(w, e3, {"e1": "g", "e2": "g", "e3": "g"},
                 scores={frozenset({"e1", "e3"}): 0.85, frozenset({"e2", "e3"}): 0.97})
    assert m is not None and m.claim_id == "c-e2" and m.similarity == 0.97


def test_group_none_when_ungrouped():
    w = World()
    w.add(rec("e1", "use pnpm"), group="e1")
    e2 = rec("e2", "use pnpm", user="u2")
    assert gm_match(w, e2, {"e1": "g"}) is None


def test_group_ignores_events_after_bound():
    w = World()
    w.add(rec("e1", "use pnpm"), group="e1")
    e2 = rec("e2", "pnpm please", user="u2")
    w.db.events.append(e2)
    w.db.events.append(rec("e3", "later", user="u3"))
    ctx = w.ctx({"e1": "g", "e2": "g", "e3": "g"}, upto=2)
    assert group_of(rec("e3", "later", user="u3"), ctx) is None
    assert group_of(e2, ctx) == "e1"


def test_copy_verbatim_both_matchers():
    for mk in ("group", "lexical"):
        w = World()
        w.add(rec("e1", LONG), group="e1")
        e2 = rec("e2", LONG, user="u2")
        if mk == "group":
            m = gm_match(w, e2, {"e1": "g", "e2": "g"})
        else:
            w.db.events.append(e2)
            m = LexicalMatcher(CFG).match(e2, w.ctx(), w.store)
        assert m is not None and m.claim_id == "c-e1" and m.copied_from == "e1"


def test_lexical_same_author_repeat():
    w = World()
    w.add(rec("e1", LONG))
    e2 = rec("e2", LONG)
    w.db.events.append(e2)
    m = LexicalMatcher(CFG).match(e2, w.ctx(), w.store)
    assert m is not None and m.claim_id == "c-e1" and m.copied_from is None


def test_lexical_one_word_different():
    w = World()
    w.add(rec("e1", LONG))
    e2 = rec("e2", LONG2, user="u2")
    w.db.events.append(e2)
    m = LexicalMatcher(CFG).match(e2, w.ctx(), w.store)
    assert m is not None and m.claim_id == "c-e1"


def test_lexical_below_threshold():
    w = World()
    w.add(rec("e1", LONG))
    e2 = rec("e2", "the deploy script restarts things", user="u2")
    w.db.events.append(e2)
    assert LexicalMatcher(CFG).match(e2, w.ctx(), w.store) is None


def test_lexical_boundaries():
    cases = [
        rec("e2", LONG, scope="personal"),
        rec("e2", LONG, team="t2"),
        rec("e2", LONG, conditions={"os": "mac"}),
    ]
    for e2 in cases:
        w = World()
        w.add(rec("e1", LONG))
        w.db.events.append(e2)
        assert LexicalMatcher(CFG).match(e2, w.ctx(), w.store) is None


def test_lexical_skips_deprecated():
    w = World()
    w.add(rec("e1", LONG), status="deprecated")
    e2 = rec("e2", LONG, user="u2")
    w.db.events.append(e2)
    assert LexicalMatcher(CFG).match(e2, w.ctx(), w.store) is None


def test_fake_neighbors_respect_visibility():
    db = InMemoryDatabase()
    db.events.append(rec("e1", "x y", scope="personal", user="u1"))
    db.events.append(rec("e2", "x y", user="u2"))
    fake = FakeSimilarity(db.events, {"e1": "g", "e2": "g"})
    caller = Caller(user="u2", team="t1", session="s", at=T0)
    assert fake.neighbors("e2", caller, "team") == []
    caller1 = Caller(user="u1", team="t1", session="s", at=T0)
    assert [e.id for e, _ in fake.neighbors("e2", caller1, "team")] == ["e1"]


def test_match_claim_without_port():
    w = World()
    w.add(rec("e1", LONG))
    e2 = rec("e2", LONG, user="u2")
    w.db.events.append(e2)
    ctx = w.ctx()
    assert similar_of(ctx) is None
    m, grp = match_claim(e2, ctx, w.store, CFG)
    assert m is not None and m.claim_id == "c-e1" and grp is None


def test_match_claim_with_port():
    w = World()
    w.add(rec("e1", "use pnpm"), group="e1")
    e2 = rec("e2", "pnpm please", user="u2")
    w.db.events.append(e2)
    ctx = w.ctx({"e1": "g", "e2": "g"})
    m, grp = match_claim(e2, ctx, w.store, CFG)
    assert m is not None and m.claim_id == "c-e1" and grp == "e1"
