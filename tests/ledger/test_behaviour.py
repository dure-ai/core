"""Ledger behaviour end to end through `Runtime`, over every store (memory,
SQLite, PostgreSQL) and with the similarity port present or absent."""

from __future__ import annotations

from contextlib import closing
from datetime import timedelta

import pytest

from cortex_mem.contract import Caller, Demote, Engine, Promote
from cortex_mem.engines.ledger import LedgerEngine
from cortex_mem.store.memory import InMemoryDatabase
from cortex_mem.store.postgres import PostgresDatabase
from cortex_mem.store.sqlite import SqliteDatabase
from tests.ledger.helpers import T0, ev, knowledge_ids, runtime, state

FACT = "the deploy script must run migrations before restarting the api server every time"
# Jaccard with FACT is 11/13: the same claim on the lexical path, not a copy.
NEAR = "the deploy script must run migrations before restarting the api server each time"
# A different wording: only a similarity port can join it to FACT.
OTHER = "always apply the database schema changes first, then restart the service"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def db(request):
    if request.param == "memory":
        yield InMemoryDatabase()
    elif request.param == "sqlite":
        yield SqliteDatabase()
    else:
        url = request.getfixturevalue("pg_schema")
        with closing(PostgresDatabase(url)) as pg:
            yield pg


def decisions_of(rt, kind):
    return [v.decision for v in rt.verdicts if isinstance(v.decision, kind)]


def judgment(id, user, *, kind="outcome", outcome="success", signal="review", refs=("e1",),
             minutes=1, scope="team", session=None):
    return ev(id, "", kind=kind, user=user, signal=signal, outcome=outcome, refs=refs,
              minutes=minutes, scope=scope, session=session)


def caller(user="u3", session="s9", minutes=10):
    return Caller(user=user, team="t1", session=session, at=T0 + timedelta(minutes=minutes))


def test_the_engine_satisfies_the_contract():
    assert isinstance(LedgerEngine(), Engine)


@pytest.mark.parametrize("port", [False, True], ids=["lexical", "port"])
def test_one_person_repeating_five_times_never_becomes_team_knowledge(db, port):
    groups = {f"e{i}": "g" for i in range(1, 6)} if port else None
    rt, _ = runtime(db=db, groups=groups)
    for i in range(1, 6):
        rt.append(ev(f"e{i}", FACT, session=f"s{i}", minutes=i))
    rt.tick(T0 + timedelta(days=1))

    assert knowledge_ids(db) == []
    assert [c.status for c in state(db).claims()] == ["candidate"]
    # A lone person's claim may be shown as a memory, never as knowledge.
    assert [h.status for h in rt.handoff("deploy migrations", caller(), "team")] == ["memory"]
    assert rt.rejections == []


def test_two_people_confirming_promote_then_verified_failures_demote(db):
    rt, _ = runtime(db=db)
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(judgment("e2", "u2", minutes=1))
    assert knowledge_ids(db) == ["c-e1"]
    assert [(h.id, h.status) for h in rt.handoff("deploy migrations", caller(), "team")] == [
        ("c-e1", "knowledge")
    ]

    # One verified failure weighs less than the two confirmations: disputed, kept.
    rt.append(judgment("e3", "u3", signal="test", outcome="failure", minutes=5))
    assert decisions_of(rt, Demote) == []
    assert state(db).claim("c-e1").status == "disputed"
    assert knowledge_ids(db) == ["c-e1"]

    rt.append(judgment("e4", "u4", signal="test", outcome="failure", minutes=6))
    [demote] = decisions_of(rt, Demote)
    assert demote.why == "invalid"
    assert knowledge_ids(db) == []
    assert "c-e1" not in [h.id for h in rt.handoff("deploy migrations", caller(), "team")]
    assert rt.rejections == []


def test_with_the_port_two_peoples_records_in_one_group_promote(db):
    rt, _ = runtime(db=db, groups={"e1": "g", "e2": "g"})
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))
    assert knowledge_ids(db) == ["c-e1"]
    assert len(decisions_of(rt, Promote)) == 1
    assert rt.rejections == []


def test_without_the_port_two_wordings_do_not_promote(db):
    rt, _ = runtime(db=db)
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", OTHER, user="u2", minutes=1))
    assert knowledge_ids(db) == []
    assert [c.status for c in state(db).claims()] == ["candidate", "candidate"]


@pytest.mark.parametrize("port", [False, True], ids=["lexical", "port"])
def test_a_personal_confirmation_is_no_evidence_for_team_knowledge(db, port):
    rt, _ = runtime(db=db, groups={"e1": "g"} if port else None)
    rt.append(ev("e1", FACT))
    rt.append(judgment("e2", "u2", kind="human_confirm", signal=None, scope="personal", minutes=1))

    assert knowledge_ids(db) == []
    assert {e.event_id for e in state(db).evidence("c-e1")} == {"e1"}
    assert rt.rejections == []


@pytest.mark.parametrize("port", [False, True], ids=["lexical", "port"])
def test_repeating_what_was_handed_adds_no_vote(db, port):
    groups = {"e1": "g", "e2": "g", "e4": "g"} if port else None
    rt, _ = runtime(db=db, groups=groups)
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))
    assert knowledge_ids(db) == ["c-e1"]

    handed = rt.recall("deploy migrations", caller("u3", "s9", 10), "team", "r1")
    assert [h.id for h in handed] == ["c-e1"]
    rt.append(ev("e4", NEAR, user="u3", session="s9", minutes=11))

    rows = {e.event_id: e for e in state(db).evidence("c-e1")}
    assert rows["e4"].derived == "retrieved"
    assert {e.key for e in rows.values() if e.derived is None} == {"u:u1", "u:u2"}
    assert rt.rejections == []


def _twenty_events():
    texts = [FACT, NEAR, OTHER]
    out = []
    for i in range(1, 15):
        out.append(ev(f"e{i}", texts[i % 3], user=f"u{i % 4 + 1}", minutes=i))
    out.append(judgment("e15", "u2", refs=("e1",), minutes=15))
    out.append(judgment("e16", "u3", refs=("e1",), minutes=16))
    out.append(judgment("e17", "u4", signal="test", outcome="failure", refs=("e2",), minutes=17))
    out.append(ev("e18", FACT, user="u5", minutes=18))
    out.append(judgment("e19", "u1", kind="human_confirm", signal=None, refs=("e3",), minutes=19))
    out.append(ev("e20", OTHER, user="u6", minutes=20))
    return out


GROUPS = {f"e{i}": "ga" if i % 3 else "gb" for i in range(1, 21)}


def _run(db, groups):
    rt, _ = runtime(db=db, groups=groups)
    for e in _twenty_events():
        rt.append(e)
    rt.sync()
    rt.tick(T0 + timedelta(days=2))
    handoff = [(h.id, h.content, h.sources, h.status)
               for h in rt.handoff("deploy migrations api server", caller(), "team")]
    return (
        [repr(v.decision) for v in rt.verdicts],
        sorted((k.id, k.title, k.body, k.sources) for k in db.knowledge.all()),
        handoff,
        state(db).skipped(),
    )


@pytest.mark.parametrize("port", [False, True], ids=["lexical", "port"])
def test_the_same_events_and_groups_give_the_same_decisions(db, port):
    groups = GROUPS if port else None
    first = _run(db, groups)
    assert first[0], "the scenario must decide something"
    assert first == _run(InMemoryDatabase(), groups)


@pytest.mark.parametrize("port", [False, True], ids=["lexical", "port"])
def test_another_state_version_rebuilds_the_same_knowledge_base(db, port):
    groups = {"e1": "g", "e2": "g", "e4": "g"} if port else None
    rt, _ = runtime(db=db, groups=groups)
    rt.append(ev("e1", FACT, user="u1"))
    rt.append(ev("e2", NEAR, user="u2", minutes=1))
    rt.append(judgment("e3", "u3", minutes=2))
    rt.append(ev("e4", NEAR, user="u3", minutes=3))
    before = sorted((k.id, k.title, k.body, k.sources) for k in db.knowledge.all())
    claims = [(c.id, c.status) for c in state(db).claims()]
    assert before

    class Next(LedgerEngine):
        state_version = "2"

    rt2, _ = runtime(engine=Next(), db=db, groups=groups)
    rt2.sync()
    after = sorted((k.id, k.title, k.body, k.sources) for k in db.knowledge.all())
    assert after == before
    assert [(c.id, c.status) for c in state(db).claims()] == claims
    assert rt2.rejections == []
