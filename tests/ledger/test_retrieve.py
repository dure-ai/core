from __future__ import annotations

from datetime import timedelta

from cortex_mem.contract import Caller, EngineContext
from cortex_mem.engines.ledger import LedgerEngine
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase
from .fakes import FakeSimilarity
from .helpers import ENGINE, T0, edit_state, ev, runtime, state
from .test_engine_judgments import FACT, NEAR, judgment

# three unrelated claims that share the query word "service"
ALPHA = "service alpha rotates signing keys quarterly and publishes new keys to every region"
BETA = "service beta invalidates cache entries after each schema change to avoid stale reads"
GAMMA = "service gamma drains queue workers gracefully before any node reboot happens"
THIRD = "the deploy script must run migrations before restarting the api server every single time"


def caller(user="u1", *, minutes=100, context=None, team="t1"):
    return Caller(
        user=user, team=team, session="sx", at=T0 + timedelta(minutes=minutes),
        context=context or {},
    )


def established(rt):
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))


def test_one_item_per_claim():
    rt, _ = runtime()
    for i in range(4):
        rt.append(ev(f"e{i + 1}", FACT, session=f"s{i}", minutes=i))
    [h] = rt.handoff("deploy migrations api server", caller(), "team")
    assert h.id == "c-e1"


def test_sources_are_all_supporting_evidence():
    rt, db = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))
    rt.append(judgment("e3", "u3", minutes=2))
    rt.append(ev("e4", FACT, user="u5", minutes=3))  # an exact copy: not a source
    rt.append(ev("e5", "", kind="recall", user="u4", session="s9", refs=("c-e1",), minutes=4))
    rt.append(ev("e6", NEAR, user="u4", session="s9", minutes=5))  # repeats what it was handed
    [h] = rt.handoff("deploy migrations api server", caller(minutes=100), "team")
    assert h.sources == ("e1", "e2", "e3")
    keys = {e.key for e in state(db).evidence("c-e1") if e.event_id in h.sources}
    assert f"{len(keys)} sources" in h.content


class _Semantic(FakeSimilarity):
    """A port that finds records by meaning: it returns what the test says."""

    def search(self, query, caller, scope, k=20, kinds=None):
        return [(e, 0.9) for e in self._log if e.id in self._groups]


class _SemanticRuntime(Runtime):
    def _ctx(self, state, upto):
        return EngineContext(
            events=self.events, knowledge=self.knowledge, state=state,
            similar=_Semantic(self.events, {"e1": "g"}, upto=upto),
        )


def test_a_claim_found_only_by_similarity_is_handed_off():
    query = "how do we ship schema updates"  # no word in common with FACT
    plain, _ = runtime()
    plain.append(ev("e1", FACT))
    assert plain.handoff(query, caller(), "team") == []

    db = InMemoryDatabase()
    rt = _SemanticRuntime(LedgerEngine(), db.events, db.knowledge, db.state(ENGINE),
                          transaction=db.transaction)
    rt.append(ev("e1", FACT))
    assert [h.id for h in rt.handoff(query, caller(), "team")] == ["c-e1"]


def test_established_outranks_a_candidate_of_equal_relevance():
    rt, _ = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))
    rt.append(ev("e3", ALPHA, user="u3", minutes=2))
    # equal relevance: each is the only hit of one of the two lists
    got = rt.handoff("service alpha", caller("u9"), "team")
    assert [h.id for h in got] == ["c-e3"]
    both = rt.handoff("service deploy", caller("u9"), "team")
    assert [h.id for h in both] == ["c-e1", "c-e3"]
    assert [h.status for h in both] == ["knowledge", "memory"]


def test_deprecated_claims_are_not_handed_off():
    rt, db = runtime()
    rt.append(ev("e1", FACT))

    def down(st):
        c = st.claim("c-e1")
        c.status = "deprecated"
        st.put_claim(c)

    edit_state(db, down)
    assert rt.handoff("deploy script migrations", caller(), "team") == []


def test_conditions_that_disagree_with_the_caller_are_filtered():
    rt, _ = runtime()
    rt.append(ev("e1", FACT, conditions={"service": "api"}))
    assert rt.handoff("deploy migrations", caller(context={"service": "web"}), "team") == []
    assert len(rt.handoff("deploy migrations", caller(context={"service": "api"}), "team")) == 1
    assert len(rt.handoff("deploy migrations", caller(), "team")) == 1
    assert len(rt.handoff("deploy migrations", caller(context={"region": "eu"}), "team")) == 1


def test_content_carries_the_provenance_line():
    rt, _ = runtime()
    established(rt)
    [h] = rt.handoff("deploy migrations", caller(), "team")
    assert h.content.startswith(FACT + "\n\n[established · observation 2 · 2 sources · since 2026-10-19]")
    assert h.status == "knowledge" and h.sources == ("e1", "e2")
    assert h.confidence is not None and 0 < h.confidence < 1


def test_provenance_lists_channels_in_order_and_disputes():
    rt, _ = runtime()
    established(rt)
    rt.append(judgment("e3", "u3", signal="test", minutes=2))
    rt.append(judgment("e4", "u4", signal="human", kind="human_confirm", minutes=3))
    [h] = rt.handoff("deploy migrations", caller(), "team")
    assert "[established · human 1, test 1, observation 2 · 4 sources" in h.content


def test_own_candidate_ranks_above_someone_elses_candidate():
    rt, _ = runtime()
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", ALPHA, user="u2", minutes=1))
    both = "deploy service"  # both match one term, so equal relevance
    assert [h.id for h in rt.handoff(both, caller("u2"), "team")] == ["c-e2", "c-e1"]
    assert [h.id for h in rt.handoff(both, caller("u1"), "team")] == ["c-e1", "c-e2"]


def test_personal_claims_of_others_never_appear():
    rt, db = runtime()
    rt.append(ev("e1", FACT, user="u1", scope="personal"))
    assert rt.handoff("deploy migrations", caller("u2"), "team") == []
    assert len(rt.handoff("deploy migrations", caller("u1"), "personal")) == 1
    # the engine itself drops it, before the runtime's filter
    ctx = EngineContext(events=rt.events, knowledge=rt.knowledge, state=db.state(ENGINE))
    assert LedgerEngine().handoff("deploy migrations", caller("u2"), "team", ctx) == []


def test_rejected_promotion_is_handed_off_as_memory():
    rt, db = runtime()
    rt.append(ev("e1", FACT))

    def fake(st):
        c = st.claim("c-e1")
        c.status, c.promoted = "established", True
        st.put_claim(c)

    edit_state(db, fake)
    [h] = rt.handoff("deploy migrations", caller(), "team")
    assert h.status == "memory"


def test_handoff_does_not_write_state():
    rt, db = runtime()
    established(rt)
    before = dict(db.state(ENGINE).scan(""))
    assert rt.handoff("deploy migrations", caller(), "team")
    assert dict(db.state(ENGINE).scan("")) == before


def test_k_limits_the_result():
    rt, _ = runtime()
    rt.append(ev("e1", FACT))
    rt.append(ev("e2", ALPHA, minutes=1))
    rt.append(ev("e3", BETA, minutes=2))
    rt.append(ev("e4", GAMMA, minutes=3))
    assert len(rt.handoff("service", caller(), "team", k=2)) == 2
    assert len(rt.handoff("service", caller(), "team", k=6)) == 3


def test_recency_is_off_by_default_and_halves_with_the_half_life():
    day = 60 * 24

    def handed(cfg, *, old, new, at_days=100):
        rt, _ = runtime(engine=LedgerEngine(cfg))
        rt.append(ev("e1", ALPHA, user="u1", minutes=old * day))
        rt.append(ev("e2", GAMMA, user="u1", minutes=new * day))
        return [h.id for h in rt.handoff("service", caller("u1", minutes=at_days * day), "team")]

    # off: swapping which claim is older changes nothing
    assert handed({}, old=0, new=90) == handed({}, old=90, new=0)
    # on: the more recent claim leads whichever it is
    assert handed({"half_life_days": "30"}, old=0, new=90)[0] == "c-e2"
    assert handed({"half_life_days": "30"}, old=90, new=0)[0] == "c-e1"
    # the floor: ages of thousands of years still leave both in play
    far = handed({"half_life_days": "30"}, old=0, new=1, at_days=10**6)
    assert sorted(far) == ["c-e1", "c-e2"]
