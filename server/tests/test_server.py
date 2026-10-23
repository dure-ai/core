import json
import sys
from datetime import UTC, datetime, timedelta

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cortex_mem.contract import Demote
from cortex_mem_server.server import ScopePolicy, Service, load_engine
from cortex_mem.store.sqlite import SqliteDatabase

T0 = datetime(2026, 10, 19, tzinfo=UTC)


def service(db, user="u1", team="t1", session="s1", policy=ScopePolicy(), engine="promote_all", repo=None):
    return Service(db, engine, user=user, team=team, session=session, repo=repo, policy=policy, clock=lambda: T0)


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
    m = service(SqliteDatabase(), policy=ScopePolicy.parse("personal", "acme/*=team"), repo="acme/api")

    assert m.record("x")["scope"] == "team"
    assert m.record("y", scope="org")["scope"] == "team"
    assert m.record("z", scope="personal")["scope"] == "personal"


def test_repo_argument_cannot_widen_configured_repo_policy():
    """PR #2 review: the agent chose the policy by naming a repo with a wider default."""
    db = SqliteDatabase()
    alice = service(db, user="alice", team="t5", policy=ScopePolicy.parse("personal", "acme/handbook=org"),
                    repo="alice/private")

    result = alice.record("사내 비밀 메모", scope="org", repo="acme/handbook")

    assert result["scope"] == "personal"
    assert db.events.get([result["id"]])[0].repo == "acme/handbook"  # still says where the evidence is
    bob = service(db, user="bob", team="t9")
    assert bob.recall("비밀 메모") == []


def test_without_configured_repo_only_the_default_policy_applies():
    m = service(SqliteDatabase(), policy=ScopePolicy.parse("personal", "acme/*=team"))

    assert m.record("x", repo="acme/api")["scope"] == "personal"


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


def test_outcome_on_team_knowledge_is_seen_by_the_team_and_demotes_it():
    """Found with real Claude Code sessions: a failure report without a repo was
    personal, so it could not count as evidence against team knowledge."""
    db = SqliteDatabase()
    policy = ScopePolicy.parse("personal", "acme/*=team")
    me = service(db, user="u1", policy=policy, engine=f"{__name__}:DemoteOnFailure", repo="acme/api")
    me.record("deploy needs approval")
    [hit] = me.recall("deploy")

    teammate = service(db, user="u2", policy=policy, engine=f"{__name__}:DemoteOnFailure")
    report = teammate.report_outcome("failure", [hit["id"]], text="approval step was removed")

    assert report["scope"] == "team"
    assert db.knowledge.all() == []


def test_outcome_scope_can_be_narrowed_and_falls_back_to_policy():
    db = SqliteDatabase()
    policy = ScopePolicy.parse("personal", "acme/*=team")
    m = service(db, policy=policy, repo="acme/api")
    m.record("deploy needs approval")
    [hit] = m.recall("deploy")
    elsewhere = service(db, policy=policy)

    assert m.report_outcome("failure", [hit["id"]], scope="personal")["scope"] == "personal"
    assert m.report_outcome("failure", ["not-knowledge"])["scope"] == "team"
    assert elsewhere.report_outcome("failure", ["not-knowledge"])["scope"] == "personal"
    assert elsewhere.report_outcome("failure", ["not-knowledge"], repo="acme/api")["scope"] == "personal"


def test_outcome_wider_than_repo_policy_keeps_the_judgement_but_not_the_text():
    """PR #2 review: a report from a personal-policy repo on org knowledge was
    shared org-wide together with the agent's words."""
    db = SqliteDatabase()
    policy = ScopePolicy.parse("personal", "acme/handbook=org")
    writer = service(db, user="u1", policy=policy, repo="acme/handbook", engine=f"{__name__}:DemoteOnFailure")
    writer.record("release notes go in CHANGELOG.md")
    [hit] = writer.recall("release notes")
    bob = service(db, user="bob", team="t9", policy=policy, repo="bob/private",
                  engine=f"{__name__}:DemoteOnFailure")

    report = bob.report_outcome("failure", [hit["id"]], text="failed in my private repo at /secret/x")

    assert (report["scope"], report["text_redacted"]) == ("org", True)
    [event] = db.events.get([report["id"]])
    assert (event.scope, event.text, event.text_redacted, event.refs) == ("org", "", True, (hit["id"],))
    assert db.knowledge.all() == []  # the judgement still demotes


def test_outcome_within_repo_policy_keeps_its_text():
    db = SqliteDatabase()
    policy = ScopePolicy.parse("personal", "acme/*=team")
    m = service(db, policy=policy, repo="acme/api")
    m.record("deploy needs approval")
    [hit] = m.recall("deploy")

    report = m.report_outcome("failure", [hit["id"]], text="approval step was removed")

    assert (report["scope"], report["text_redacted"]) == ("team", False)
    assert db.events.get([report["id"]])[0].text == "approval step was removed"


def test_configured_repo_applies_scope_policy_without_the_agent_passing_it():
    db = SqliteDatabase()
    m = service(db, policy=ScopePolicy.parse("personal", "acme/*=team"), repo="acme/api")

    recorded = m.record("deploy needs approval")
    elsewhere = m.record("my dotfiles use zsh", repo="me/dotfiles")

    assert recorded["scope"] == "team" and elsewhere["scope"] == "team"  # the policy follows the configuration
    assert [e.repo for e in db.events.get([recorded["id"], elsewhere["id"]])] == ["acme/api", "me/dotfiles"]


def test_recall_passes_where_the_question_was_asked():
    db = SqliteDatabase()
    m = service(db, repo="acme/api")
    m.record("deploy uses release tags")

    m.recall("deploy tags", context={"service": "billing"})
    m.recall("deploy tags", repo="acme/web")

    first, second = [e for _, e in db.events.since(0) if e.kind == "recall"]
    assert (first.repo, dict(first.conditions)) == ("acme/api", {"service": "billing"})
    assert (second.repo, dict(second.conditions)) == ("acme/web", {})


def test_outcome_without_refs_is_refused():
    m = service(SqliteDatabase())

    with pytest.raises(ValueError, match="needs refs"):
        m.report_outcome("failure", [])


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
    me = service(db, user="u1", policy=policy, repo="acme/api")
    me.record("deploy needs approval")
    me.record("my deploy shortcut", scope="personal")
    teammate = service(db, user="u2", policy=policy)
    stranger = service(db, user="u3", team="t2", policy=policy)

    assert [h["content"] for h in teammate.recall("deploy")] == ["deploy needs approval"]
    assert stranger.recall("deploy") == []


class DemoteOnFailure:
    """PromoteAll, plus: a failure outcome demotes the knowledge it refers to."""

    def __init__(self, config):
        from cortex_mem.engines.baseline import PromoteAllEngine

        self.inner = PromoteAllEngine(config)

    def observe(self, event, ctx):
        if event.kind == "outcome" and event.outcome == "failure":
            return [Demote(id, "reported failure", (event.id,)) for id in event.refs]
        return self.inner.observe(event, ctx)

    def tick(self, now, ctx):
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return self.inner.handoff(query, caller, scope, ctx, k)


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


class FailsOnBoom:
    """PromoteAll, except that any event whose text has "boom" always fails."""

    def __init__(self, config):
        from cortex_mem.engines.baseline import PromoteAllEngine

        self.inner = PromoteAllEngine(config)

    def observe(self, event, ctx):
        if "boom" in event.text:
            raise RuntimeError("engine bug")
        return self.inner.observe(event, ctx)

    def tick(self, now, ctx):
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return self.inner.handoff(query, caller, scope, ctx, k)


def test_engine_failure_does_not_block_tools_and_the_bad_event_is_skipped(caplog):
    """PR #2 review: one event the engine always fails on stopped every tool, and
    a stored record came back as an error, so the agent recorded it again."""
    db = SqliteDatabase()
    m = Service(db, f"{__name__}:FailsOnBoom", user="u1", team="t1", session="s1",
                clock=lambda: T0, max_failures=3)

    bad = m.record("boom")
    assert bad["delivered"] is False and db.events.get([bad["id"]])  # stored, not lost

    good = m.record("deploy uses release tags")  # attempts 2 and 3 on "boom", then skipped

    assert good["delivered"] is True
    assert [h["content"] for h in m.recall("deploy tags")] == ["deploy uses release tags"]
    assert [s.event_id for s in m.runtime.skipped] == [bad["id"]]
    assert f"skipped event {bad['id']} after 3 failed deliveries" in caplog.text


def test_without_max_failures_the_service_keeps_retrying():
    db = SqliteDatabase()
    m = Service(db, f"{__name__}:FailsOnBoom", user="u1", team="t1", session="s1", clock=lambda: T0)

    m.record("boom")
    later = m.record("deploy uses release tags")

    assert later["delivered"] is False and m.runtime.skipped == []
    assert m.runtime.state.cursor() == 0


def test_refused_event_is_still_an_error():
    m = Service(SqliteDatabase(), f"{__name__}:FailsOnBoom", user="u1", team="t1", session="s1",
                clock=lambda: T0, max_failures=3)

    with pytest.raises(ValueError, match="needs refs"):
        m.report_outcome("failure", [])


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


def call(db, tool, args, user="u1", raw=False):
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "cortex_mem_server.server"],
            env={"CORTEX_DB_PATH": str(db), "CORTEX_USER": user, "CORTEX_TEAM": "t1"},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result if raw else json.loads(result.content[0].text)

    return anyio.run(run)


def test_stdio_session_b_recalls_what_session_a_recorded(tmp_path):
    db = tmp_path / "cortex.db"
    recorded = call(db, "record_episode", {"text": "cloud sql needs the auth proxy"})
    assert recorded["scope"] == "personal"

    hits = call(db, "recall", {"query": "cloud sql proxy"})["memories"]

    assert [h["content"] for h in hits] == ["cloud sql needs the auth proxy"]


def test_stdio_outcome_without_refs_tells_the_agent_why(tmp_path):
    result = call(tmp_path / "cortex.db", "report_outcome", {"outcome": "failure", "refs": []}, raw=True)

    assert result.is_error and "needs refs" in result.content[0].text
