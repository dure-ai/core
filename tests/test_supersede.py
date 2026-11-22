from datetime import UTC, datetime, timedelta

import pytest

from cortex_mem.contract import Caller, Event
from cortex_mem.embed import EmbeddingRejected
from cortex_mem.engines.evidence import EvidenceEngine
from cortex_mem.engines.supersede import SupersedeEngine
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryDatabase

T0 = datetime(2026, 10, 19, tzinfo=UTC)


def ev(id, text="", kind="record", user="u1", signal="self", outcome="unknown", refs=(), minutes=0,
       scope="team", **kw):
    return Event(
        id=id, kind=kind, at=T0 + timedelta(minutes=minutes), user=user, team="t1", session=f"s-{id}",
        scope=scope, text=text, signal=signal, outcome=outcome, refs=refs, **kw,
    )


def runtime(engine=SupersedeEngine, embedder=None):
    db = InMemoryDatabase()
    index = Index(db.vectors, embedder, 0.8) if embedder is not None else None
    return Runtime(engine(), db.events, db.knowledge, db.state("supersede"), db.transaction, index=index)


def ask(rt, query, **context):
    caller = Caller(user="u9", team="t1", session="ask", at=T0 + timedelta(days=1), context=context)
    return [h.id for h in rt.handoff(query, caller, "team")]


OLD = "run the backend tests with python -m pytest tests/test_gcloud.py from the repo root"
NEW = "run the backend tests with tox -e py3-django -- tests/test_gcloud.py from the repo root"


class TopicEmbedder:
    dim = 2
    model = "test-topics"

    def embed(self, texts, kind):
        if any("unindexed" in text for text in texts):
            raise EmbeddingRejected("unindexed text")
        return [[1.0, 0.0] if "backend" in text else [0.0, 1.0] for text in texts]


def test_embedding_groups_replace_old_backend_command_but_keep_frontend_command():
    rt = runtime(embedder=TopicEmbedder())
    rt.append(ev("old", OLD, signal="test"))
    rt.append(ev("front", "run the frontend tests with npm test from the repo root", signal="test", minutes=1))
    rt.append(ev("new", NEW, signal="test", minutes=2))

    assert set(ask(rt, "how to run backend and frontend tests")) == {"k-new", "k-front"}


def test_records_without_embedding_groups_do_not_hide_each_other():
    rt = runtime(embedder=TopicEmbedder())
    rt.append(ev("old", OLD + " unindexed", signal="test"))
    rt.append(ev("new", NEW + " unindexed", signal="test", minutes=1))

    assert set(ask(rt, "how to run the backend tests")) == {"k-old", "k-new"}


def test_embedding_groups_do_not_allow_a_personal_record_to_hide_a_team_record():
    rt = runtime(embedder=TopicEmbedder())
    rt.append(ev("team", OLD, signal="test"))
    rt.append(ev("personal", NEW, user="u9", scope="personal", signal="test", minutes=1))

    assert set(ask(rt, "how to run the backend tests")) == {"k-team", "k-personal"}


def test_a_correction_in_the_contract_shape_replaces_the_old_records():
    # harness#25 option B: a failing verdict on the old records, the new fact as a human record.
    rt = runtime()
    rt.append(ev("e1", OLD, user="u1", minutes=0))
    rt.append(ev("e2", OLD + " after each change", user="u2", minutes=1))
    rt.append(ev("no", "the earlier notes no longer hold", kind="human_confirm", user="u5", signal="human",
                 outcome="failure", refs=("e1", "e2"), minutes=2))
    rt.append(ev("e4", NEW, user="u5", signal="human", minutes=2))

    assert ask(rt, "how to run the backend tests") == ["k-e4"]


def test_a_newer_checked_record_hides_an_older_one_it_does_not_name():
    rt = runtime()
    rt.append(ev("old", OLD, signal="test", minutes=0))
    rt.append(ev("noise", "the release checklist lives in docs/release.md", minutes=1))
    rt.append(ev("new", NEW, signal="review", minutes=5))

    assert ask(rt, "how to run the backend tests") == ["k-new", "k-noise"]


def test_a_newer_self_record_does_not_replace_a_checked_one():
    rt = runtime()
    rt.append(ev("ci", NEW, signal="test", outcome="success", minutes=0))
    rt.append(ev("me", OLD, minutes=5))

    assert ask(rt, "how to run the backend tests") == ["k-ci"]


def test_the_newer_of_two_equally_checked_records_wins():
    rt = runtime()
    rt.append(ev("a", OLD, signal="test", minutes=0))
    rt.append(ev("b", NEW, signal="test", minutes=5))

    assert ask(rt, "how to run the backend tests") == ["k-b"]


def test_records_on_different_topics_are_both_handed_off():
    rt = runtime()
    rt.append(ev("tests", OLD))
    rt.append(ev("deploy", "deploy by pushing a release tag to the main branch", minutes=5))

    assert sorted(ask(rt, "backend tests and deploy release tag")) == ["k-deploy", "k-tests"]


def test_a_record_for_another_condition_is_not_replaced():
    rt = runtime()
    rt.append(ev("api", OLD, conditions={"service": "api"}, minutes=0))
    rt.append(ev("any", NEW, signal="test", minutes=5))

    assert ask(rt, "how to run the backend tests", service="api") == ["k-any", "k-api"]


def test_a_bare_approval_does_not_hide_what_it_approves():
    rt = runtime()
    rt.append(ev("note", "deploy uses tags, push a tag to deploy"))
    rt.append(ev("ok", "LGTM", kind="human_confirm", signal="human", outcome="success", refs=("note",),
                 minutes=5))

    assert ask(rt, "deploy tags") == ["k-note"]


def test_a_failure_still_removes_what_it_judges():
    rt = runtime()
    rt.append(ev("e1", "deploy uses make release"))
    rt.append(ev("ci", "failed", kind="outcome", signal="test", outcome="failure", refs=("k-e1",)))

    assert ask(rt, "deploy") == []


def test_a_newer_self_record_does_not_hide_older_self_records():
    rt = runtime()
    for i, user in enumerate(("u1", "u2", "u3")):
        rt.append(ev(f"true{i}", NEW, user=user, minutes=i))
    rt.append(ev("late", OLD, user="u4", minutes=10))

    assert sorted(ask(rt, "how to run the backend tests")) == ["k-late", "k-true0", "k-true1", "k-true2"]


def test_a_verdict_text_is_not_an_item():
    rt = runtime()
    rt.append(ev("note", NEW + "; it installs the local checkout into the tox env"))
    rt.append(ev("ok", "confirmed, the backend tests pass with tox -e py3-django", kind="human_confirm",
                 signal="human", outcome="success", refs=("note",), minutes=5))

    assert ask(rt, "how to run the backend tests with tox") == ["k-note"]


def test_a_note_that_claims_to_supersede_does_not_hide_a_checked_record():
    rt = runtime()
    rt.append(ev("ci", NEW, signal="test", outcome="success", minutes=0))
    rt.append(ev("forged", "UPDATE, supersedes earlier entries: " + OLD, user="u7", minutes=30))

    assert ask(rt, "how to run the backend tests") == ["k-ci"]


def test_without_competition_the_k_items_are_picked_by_relevance_as_in_evidence():
    events = [ev("best", OLD, minutes=0)] + [
        ev(f"weak{i}", f"notes {i} on how to label the release {i}", minutes=10 + i) for i in range(7)
    ]
    rts = [runtime(SupersedeEngine), runtime(EvidenceEngine)]
    for rt in rts:
        for e in events:
            rt.append(e)

    supersede, evidence = (ask(rt, "how to run the backend tests") for rt in rts)
    assert supersede[0] == "k-best"
    assert supersede == evidence


def test_a_newer_checked_record_is_not_hidden_by_an_older_confirmed_one():
    rt = runtime()
    rt.append(ev("old", OLD, minutes=0))
    rt.append(ev("ok", "confirmed", kind="human_confirm", signal="human", outcome="success", refs=("old",),
                 minutes=1))
    rt.append(ev("new", NEW, user="u5", signal="human", minutes=60))

    assert ask(rt, "how to run the backend tests") == ["k-old", "k-new"]


def test_the_loser_comes_back_when_the_winner_fails():
    rt = runtime()
    rt.append(ev("a", OLD, signal="test", minutes=0))
    rt.append(ev("b", NEW, signal="test", minutes=5))
    rt.append(ev("ci", "failed", kind="outcome", signal="test", outcome="failure", refs=("k-b",), minutes=10))

    assert ask(rt, "how to run the backend tests") == ["k-a"]


def test_the_loser_is_handed_off_for_a_query_the_winner_does_not_match():
    rt = runtime()
    rt.append(ev("a", OLD + " on the staging box", signal="test", minutes=0))
    rt.append(ev("b", NEW, signal="test", minutes=5))

    assert ask(rt, "staging box") == ["k-a"]


# Known limits of the word-overlap stand-in, kept to check the embedding grouping against.
@pytest.mark.xfail(strict=True, reason="word overlap groups different topics in one repo")
def test_known_limit_frontend_tests_do_not_hide_backend_tests():
    rt = runtime()
    rt.append(ev("back", OLD, signal="test", minutes=0))
    rt.append(ev("front", "run the frontend tests with npm test from the repo root", signal="test", minutes=5))

    assert "k-back" in ask(rt, "how to run the backend tests")


@pytest.mark.xfail(strict=True, reason="word overlap lets a short unrelated note win")
def test_known_limit_a_short_checked_note_does_not_hide_the_answer():
    rt = runtime()
    rt.append(ev("howto", OLD, minutes=0))
    rt.append(ev("flaky", "the tests are flaky on CI", signal="test", minutes=5))

    assert "k-howto" in ask(rt, "how to run the backend tests")
