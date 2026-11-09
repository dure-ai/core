from datetime import UTC, datetime, timedelta

from cortex_mem.contract import Caller, Event
from cortex_mem.engines.evidence import EvidenceEngine
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase

T0 = datetime(2026, 10, 19, tzinfo=UTC)


def ev(id, text="", kind="record", user="u1", signal="self", outcome="unknown", refs=(), minutes=0, **kw):
    return Event(
        id=id, kind=kind, at=T0 + timedelta(minutes=minutes), user=user, team="t1", session=f"s-{id}",
        scope="team", text=text, signal=signal, outcome=outcome, refs=refs, **kw,
    )


def runtime():
    db = InMemoryDatabase()
    return Runtime(EvidenceEngine(), db.events, db.knowledge, db.state("evidence"), db.transaction)


def ask(rt, query, **context):
    caller = Caller(user="u9", team="t1", session="ask", at=T0 + timedelta(days=1), context=context)
    return [h.id for h in rt.handoff(query, caller, "team")]


def test_every_record_is_handed_off():
    rt = runtime()
    rt.append(ev("e1", "deploy uses make release"))

    assert ask(rt, "how to deploy") == ["k-e1"]


def test_a_record_with_an_outside_signal_ranks_above_self_records():
    rt = runtime()
    rt.append(ev("self", "deploy uses make release, run make release from the root"))
    rt.append(ev("ci", "deploy works with tags", signal="test", outcome="success"))

    assert ask(rt, "deploy make release") == ["k-ci", "k-self"]


def test_a_passing_outcome_lifts_the_records_it_judges():
    rt = runtime()
    rt.append(ev("wrong", "deploy uses make release, run make release from the root"))
    rt.append(ev("right", "deploy uses tags"))
    rt.append(ev("ci", "passed", kind="outcome", signal="test", outcome="success", refs=("right",)))

    assert ask(rt, "deploy make release") == ["k-right", "k-wrong"]


def test_a_human_confirmation_replaces_the_records_it_reviews():
    rt = runtime()
    rt.append(ev("old", "schema lives in struct.py"))
    rt.append(ev("review", "after the merge the schema lives in models.py", kind="human_confirm",
                 signal="human", outcome="success", refs=("old",), minutes=1))

    [handoff] = rt.handoff("where does the schema live", Caller(user="u9", team="t1", session="a", at=T0), "team")
    assert handoff.id == "k-review"
    assert handoff.sources == ("review",)  # the reviewed note is not cited as support


def test_a_failure_removes_what_it_judges():
    rt = runtime()
    rt.append(ev("e1", "deploy uses make release"))
    rt.append(ev("ci", "failed", kind="outcome", signal="test", outcome="failure", refs=("k-e1",)))

    assert ask(rt, "deploy") == []


def test_items_for_another_condition_are_not_handed_off():
    rt = runtime()
    rt.append(ev("api", "deploy with make release", conditions={"service": "api"}))
    rt.append(ev("web", "deploy with npm run release", conditions={"service": "web"}))
    rt.append(ev("any", "deploy needs a tag"))

    assert sorted(ask(rt, "deploy release tag", service="api")) == ["k-any", "k-api"]
