import json
import sys
from datetime import UTC, datetime, timedelta

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cortex_mem_server.server import ScopePolicy, Service, load_engine
from cortex_mem_server.sqlite import SqliteDatabase

T0 = datetime(2026, 10, 19, tzinfo=UTC)


def service(db, user="u1", team="t1", session="s1", policy=ScopePolicy(), engine="promote_all"):
    return Service(db, engine, user=user, team=team, session=session, policy=policy, clock=lambda: T0)


# ── scope policy ────────────────────────────────────────────────────────────


def test_policy_uses_first_matching_repo_then_default():
    policy = ScopePolicy.parse("personal", "acme/handbook=org, acme/*=team")

    assert policy.default_for("acme/handbook") == "org"
    assert policy.default_for("acme/api") == "team"
    assert policy.default_for("other/repo") == "personal"
    assert policy.default_for(None) == "personal"


def test_agent_may_narrow_but_not_widen():
    policy = ScopePolicy.parse("personal", "acme/*=team")

    assert policy.resolve(None, "acme/api") == "team"
    assert policy.resolve("personal", "acme/api") == "personal"
    assert policy.resolve("org", "acme/api") == "team"
    assert policy.resolve("team", "other/repo") == "personal"


@pytest.mark.parametrize("default, repos", [("everyone", ""), ("personal", "acme/*=public"), ("personal", "=team")])
def test_bad_policy_is_refused(default, repos):
    with pytest.raises(ValueError):
        ScopePolicy.parse(default, repos)


def test_engine_aliases_and_paths_name_the_same_state():
    assert load_engine("promote_all")[0] == load_engine("cortex_mem.engines.baseline:PromoteAllEngine")[0]
    with pytest.raises(ValueError):
        load_engine("promote-all")


# ── service ─────────────────────────────────────────────────────────────────


def test_recall_returns_recorded_episode():
    m = service(SqliteDatabase())
    m.record("deploy uses release tags")

    assert [h["content"] for h in m.recall("deploy tags")] == ["deploy uses release tags"]


def test_record_reports_the_scope_actually_used():
    m = service(SqliteDatabase(), policy=ScopePolicy.parse("personal", "acme/*=team"))

    assert m.record("x", repo="acme/api")["scope"] == "team"
    assert m.record("y", scope="org", repo="acme/api")["scope"] == "team"
    assert m.record("z", scope="personal", repo="acme/api")["scope"] == "personal"


def test_recall_is_recorded_as_personal_recall_event_with_refs():
    db = SqliteDatabase()
    m = service(db)
    m.record("deploy uses release tags")

    hits = m.recall("deploy tags")

    recall = [e for _, e in db.events.since(0) if e.kind == "recall"]
    assert [(e.text, e.refs, e.scope) for e in recall] == [("deploy tags", tuple(h["id"] for h in hits), "personal")]


def test_report_outcome_refs_judged_ids():
    db = SqliteDatabase()
    m = service(db)
    rid = m.record("deploy uses release tags")["id"]

    oid = m.report_outcome("failure", [rid], text="tag missing", signal="test")["id"]

    [event] = db.events.get([oid])
    assert (event.kind, event.outcome, event.signal, event.refs) == ("outcome", "failure", "test", (rid,))


def test_session_started_earlier_sees_later_record(tmp_path):
    """The PR #2 review bug: B started before A recorded, and still finds it."""
    path = tmp_path / "cortex.db"
    b = service(SqliteDatabase(path), session="b")
    a = service(SqliteDatabase(path), session="a")

    a.record("cloud sql needs the auth proxy")

    assert [h["content"] for h in b.recall("cloud sql proxy")] == ["cloud sql needs the auth proxy"]


def test_teammate_sees_team_records_but_not_personal_ones():
    db = SqliteDatabase()
    policy = ScopePolicy.parse("personal", "acme/*=team")
    me = service(db, user="u1", policy=policy)
    me.record("deploy needs approval", repo="acme/api")
    me.record("my deploy shortcut", scope="personal", repo="acme/api")
    teammate = service(db, user="u2", policy=policy)
    stranger = service(db, user="u3", team="t2", policy=policy)

    assert [h["content"] for h in teammate.recall("deploy")] == ["deploy needs approval"]
    assert stranger.recall("deploy") == []


class ClockedEngine:
    """Records every tick it gets."""

    ticks: list[datetime] = []

    def __init__(self, config):
        pass

    def observe(self, event, ctx):
        return []

    def tick(self, now, ctx):
        self.ticks.append(now)
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return []


def test_each_call_lets_time_pass_for_the_engine():
    ClockedEngine.ticks = []
    now = [T0]
    m = Service(SqliteDatabase(), f"{__name__}:ClockedEngine", user="u1", team="t1", session="s1",
                clock=lambda: now[0])
    m.record("a")
    now[0] = T0 + timedelta(days=1)
    m.recall("a")

    assert ClockedEngine.ticks == [T0, T0 + timedelta(days=1)]


# ── stdio ───────────────────────────────────────────────────────────────────


def call(db, tool, args, user="u1"):
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "cortex_mem_server.server"],
            env={"CORTEX_DB_PATH": str(db), "CORTEX_USER": user, "CORTEX_TEAM": "t1"},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return json.loads(result.content[0].text)

    return anyio.run(run)


def test_stdio_session_b_recalls_what_session_a_recorded(tmp_path):
    db = tmp_path / "cortex.db"
    recorded = call(db, "record_episode", {"text": "cloud sql needs the auth proxy"})
    assert recorded["scope"] == "personal"

    hits = call(db, "recall", {"query": "cloud sql proxy"})["memories"]

    assert [h["content"] for h in hits] == ["cloud sql needs the auth proxy"]
