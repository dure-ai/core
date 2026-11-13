import json
import os
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


@pytest.mark.parametrize(
    "settings, message",
    [({}, "set CORTEX_TEAM"), ({"CORTEX_TEAM": "t1", "CORTEX_MAX_FAILURES": "0"}, "CORTEX_MAX_FAILURES")],
)
def test_server_refuses_to_start_with_bad_settings(tmp_path, settings, message):
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("CORTEX_")}
    env.update(settings, CORTEX_DB_PATH=str(tmp_path / "cortex.db"))
    result = subprocess.run(
        [sys.executable, "-m", "cortex_mem_server.server"], env=env, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=30, check=False,
    )

    assert result.returncode != 0 and message in result.stderr


# ── a record that replaces existing knowledge ───────────────────────────────

TIERS = "cortex_mem.engines.tiers:TierEngine"
TEAM = ScopePolicy.parse("team")
OLD_WAY = "when the test run hangs, pass --no-cache to the runner"
NEW_WAY = "the test run hang is fixed by taking the cache lock before the index lock"


def knowledge_then_replacement(db):
    """u1's note is checked knowledge; u2 records something on the same topic."""
    author, ci, fixer = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS) for u in ("u1", "ci", "u2"))
    old = author.record(OLD_WAY)
    ci.report_outcome("success", [old["id"]], signal="test")
    return fixer, ci, fixer.record(NEW_WAY)


def test_record_hands_back_related_knowledge_and_says_what_to_do():
    fixer, _, new = knowledge_then_replacement(SqliteDatabase())

    [related] = new["related"]
    assert OLD_WAY in related["content"]
    assert f"replaced_by='{new['id']}'" in new["next"]


def test_record_with_nothing_related_asks_nothing():
    m = service(SqliteDatabase(), policy=TEAM, engine=TIERS)

    assert "related" not in m.record(OLD_WAY)  # its own note is not "existing knowledge"


def test_replacement_takes_effect_only_once_the_new_record_is_checked():
    db = SqliteDatabase()
    fixer, ci, new = knowledge_then_replacement(db)
    old_id = new["related"][0]["id"]

    out = fixer.report_outcome("failure", [old_id], replaced_by=new["id"])
    assert out["restated"]
    assert [m["id"] for m in fixer.recall("test run hang") if m["status"] == "knowledge"] == [old_id]

    ci.report_outcome("success", [new["id"]], signal="test")
    knowledge = [m for m in fixer.recall("test run hang") if m["status"] == "knowledge"]
    assert len(knowledge) == 1 and NEW_WAY in knowledge[0]["content"]


def test_a_failure_report_without_replaced_by_replaces_nothing():
    db = SqliteDatabase()
    fixer, ci, new = knowledge_then_replacement(db)
    old_id = new["related"][0]["id"]

    fixer.report_outcome("failure", [old_id])
    ci.report_outcome("success", [new["id"]], signal="test")

    assert len([m for m in fixer.recall("test run hang") if m["status"] == "knowledge"]) == 2


def test_replaced_by_must_be_your_own_record_and_a_failure():
    db = SqliteDatabase()
    fixer, ci, new = knowledge_then_replacement(db)
    old_id = new["related"][0]["id"]

    with pytest.raises(ValueError, match="record you wrote"):
        ci.report_outcome("failure", [old_id], replaced_by=new["id"])
    with pytest.raises(ValueError, match="failure"):
        fixer.report_outcome("success", [old_id], replaced_by=new["id"])


def test_stdio_agent_cannot_vouch_for_itself_as_a_test(tmp_path):
    db = tmp_path / "cortex.db"
    recorded = call(db, "record_episode", {"text": "cloud sql needs the auth proxy"})

    call(db, "report_outcome", {"outcome": "failure", "refs": [recorded["id"]], "signal": "test"}, raw=True)

    signals = [e.signal for _, e in SqliteDatabase(db).events.since(0) if e.kind == "outcome"]
    assert signals in ([], ["self"])  # refused as an unknown argument, or stored as the agent's own word


# ── evidence that is not an agent's word: the cortex-mem command ────────────

import subprocess

from cortex_mem_server.cli import commit_records, confirm_commits, review


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=cwd, check=True,
                   capture_output=True)


def commit(cwd, message):
    git(cwd, "commit", "--allow-empty", "-m", message)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_record_names_the_trailer_to_put_in_a_commit():
    m = service(SqliteDatabase(), policy=TEAM, engine=TIERS)

    r = m.record(NEW_WAY)

    assert r["commit_trailer"] == f"Cortex-Record: {r['id']}"


def test_ci_reports_the_test_run_for_records_named_in_commit_trailers(tmp_path):
    db = SqliteDatabase()
    agent, ci = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS, repo="acme/loader") for u in ("u2", "ci"))
    note = agent.record(NEW_WAY)
    git(tmp_path, "init", "-q")
    commit(tmp_path, "start")
    sha = commit(tmp_path, f"Fix the cache key\n\n{note['commit_trailer']}\nCortex-Record: no-such-record")

    result = confirm_commits(ci, "HEAD~1..HEAD", "success", cwd=str(tmp_path))

    assert commit_records("HEAD~1..HEAD", str(tmp_path)) == [(sha, [note["id"], "no-such-record"])]
    assert [(r["commit"], r["records"]) for r in result["reported"]] == [(sha, [note["id"]])]
    assert result["unknown_records"] == ["no-such-record"]
    [outcome] = [e for _, e in db.events.since(0) if e.kind == "outcome"]
    assert (outcome.signal, outcome.rev, outcome.scope) == ("test", sha, "team")
    assert [m["status"] for m in agent.recall("test run hang")] == ["knowledge"]


def test_a_failed_run_takes_the_knowledge_back(tmp_path):
    db = SqliteDatabase()
    agent, ci = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS) for u in ("u2", "ci"))
    note = agent.record(NEW_WAY)
    git(tmp_path, "init", "-q")
    commit(tmp_path, f"Fix\n\n{note['commit_trailer']}")
    confirm_commits(ci, "HEAD", "success", cwd=str(tmp_path))
    commit(tmp_path, f"Rework\n\n{note['commit_trailer']}")

    confirm_commits(ci, "HEAD~1..HEAD", "failure", cwd=str(tmp_path))

    assert agent.recall("test run hang") == []


def test_a_person_approves_or_rejects_a_record():
    db = SqliteDatabase()
    agent, lead = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS) for u in ("u2", "lead"))
    note = agent.record(NEW_WAY)

    review(lead, note["id"], "approve")
    assert [m["status"] for m in agent.recall("test run hang")] == ["knowledge"]

    review(lead, note["id"], "reject", "does not hold on the release branch")
    [correction] = agent.recall("test run hang")  # the agent was handed it a moment ago, so it is told
    assert correction["status"] == "invalidated" and "release branch" in correction["content"]


def test_approving_your_own_record_does_nothing():
    m = service(SqliteDatabase(), policy=TEAM, engine=TIERS)
    note = m.record(NEW_WAY)

    review(m, note["id"], "approve")

    assert [h["status"] for h in m.recall("test run hang")] == ["memory"]


def test_a_failure_report_with_no_record_behind_it_asks_for_one():
    db = SqliteDatabase()
    author, ci, fixer = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS) for u in ("u1", "ci", "u2"))
    old = author.record(OLD_WAY)
    ci.report_outcome("success", [old["id"]], signal="test")
    [handed] = fixer.recall("test run hang")

    out = fixer.report_outcome("failure", [handed["id"]], text="only hides the bug; the cause is the lock order")
    assert "record_episode" in out["next"] and handed["id"] in out["next"]

    new = fixer.record(NEW_WAY, refs=[handed["id"]])  # following the hint is enough: no further call
    ci.report_outcome("success", [new["id"]], signal="test")
    knowledge = [m for m in fixer.recall("test run hang") if m["status"] == "knowledge"]
    assert len(knowledge) == 1 and NEW_WAY in knowledge[0]["content"]


def test_a_failure_report_after_citing_the_memory_asks_nothing():
    db = SqliteDatabase()
    fixer, _, new = knowledge_then_replacement(db)
    old_id = new["related"][0]["id"]
    fixer.record("another note on the same thing", refs=[old_id])

    assert "next" not in fixer.report_outcome("failure", [old_id])
    assert "next" not in fixer.report_outcome("success", [old_id])


# ── what was handed over before a commit, with no help from the agent ───────

from cortex_mem_server.cli import add_recalled, install_hook, to_review


def handed_then_committed(tmp_path, message="Take the cache lock before the index lock\n\nFixes the test run hang."):
    """u1's workaround is knowledge; u2 recalls it, records nothing, and commits a fix."""
    db = SqliteDatabase()
    author, ci, fixer, lead = (service(db, user=u, session=f"s-{u}", policy=TEAM, engine=TIERS, repo="acme/app")
                               for u in ("u1", "ci", "u2", "lead"))
    old = author.record(OLD_WAY)
    ci.report_outcome("success", [old["id"]], signal="test")
    git(tmp_path, "init", "-q")
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "--allow-empty", "-m",
                    "start"], cwd=tmp_path, check=True, capture_output=True,
                   env={**os.environ, "GIT_COMMITTER_DATE": "2026-10-01T00:00:00+00:00"})  # before the test clock
    [handed] = fixer.recall("test run hang")
    msg = tmp_path / "MSG"
    msg.write_text(message + "\n")
    add_recalled(fixer, str(msg), cwd=str(tmp_path))
    sha = commit(tmp_path, msg.read_text())
    return db, fixer, lead, handed["id"], sha


def test_the_hook_names_what_the_committer_was_handed_since_the_last_commit(tmp_path):
    db, fixer, lead, old_id, sha = handed_then_committed(tmp_path)

    review_list = to_review(lead, "HEAD~1..HEAD", cwd=str(tmp_path))

    [item] = review_list["to_review"]
    assert item["id"] == old_id and OLD_WAY in item["content"]
    assert [c["commit"] for c in item["commits"]] == [sha]
    msg = tmp_path / "MSG2"
    msg.write_text("unrelated follow-up\n")
    assert add_recalled(fixer, str(msg), cwd=str(tmp_path)) == []  # nothing handed since that commit


def test_someone_elses_recalls_and_other_repos_are_not_named(tmp_path):
    db, fixer, lead, old_id, _ = handed_then_committed(tmp_path)
    elsewhere = service(db, user="u2", session="s-x", policy=TEAM, engine=TIERS, repo="acme/other")
    elsewhere.recall("test run hang")
    lead.recall("test run hang")
    commit(tmp_path, "later")

    assert fixer.recalled() == [old_id]
    assert fixer.recalled(since=datetime.now(UTC) + timedelta(days=1)) == []


def test_a_repo_argument_the_agent_made_up_does_not_hide_the_recall_from_the_hook(tmp_path):
    db, fixer, lead, old_id, _ = handed_then_committed(tmp_path)
    since = datetime.now(UTC) + timedelta(days=1)  # past everything handed so far
    other, again = (service(db, user="u2", session=f"s-{r}", policy=TEAM, engine=TIERS, repo=f"acme/{r}")
                    for r in ("other", "app"))
    other.clock = again.clock = lambda: since + timedelta(hours=1)

    other.recall("test run hang", repo="acme/app")  # asked from another project: not this repo's commit
    assert fixer.recalled(since=since) == []
    assert [m["id"] for m in again.recall("test run hang", repo="T5-app")] == [old_id]

    assert fixer.recalled(since=since) == [old_id]
    asked_in = [e.repo for _, e in db.events.since(0) if e.kind == "recall" and e.at > since]
    assert asked_in == ["acme/app", "T5-app"]  # the events still say where the agent said it asked


def test_amending_the_commit_keeps_what_it_named(tmp_path, monkeypatch):
    db, fixer, lead, old_id, sha = handed_then_committed(tmp_path)
    msg = tmp_path / "MSG2"
    msg.write_text("Reworded fix\n")
    authored = subprocess.run(["git", "log", "-1", "--format=%at"], cwd=tmp_path, check=True, capture_output=True,
                              text=True).stdout.strip()
    monkeypatch.setenv("GIT_AUTHOR_DATE", f"@{authored} +0000")

    assert add_recalled(fixer, str(msg), cwd=str(tmp_path)) == [old_id]  # as git runs the hook for --amend -m

    monkeypatch.setenv("GIT_AUTHOR_DATE", "@1 +0000")
    assert add_recalled(fixer, str(msg), cwd=str(tmp_path)) == []  # a new commit on top names nothing new


def test_a_reviewer_replaces_stale_knowledge_with_the_commit(tmp_path):
    db, fixer, lead, old_id, sha = handed_then_committed(tmp_path)

    out = review(lead, old_id, "replace", commit="HEAD", cwd=str(tmp_path))

    assert out["commit"] == sha
    [now] = [m for m in service(db, user="u3", session="s-u3", policy=TEAM, engine=TIERS, repo="acme/app")
             .recall("test run hang") if m["status"] == "knowledge"]
    assert "cache lock before the index lock" in now["content"] and "Cortex-Recalled" not in now["content"]
    assert to_review(lead, "HEAD~1..HEAD", cwd=str(tmp_path)) == {"to_review": [], "no_longer_knowledge": [old_id]}


def test_a_reviewer_keeps_knowledge_that_still_holds(tmp_path):
    db, fixer, lead, old_id, _ = handed_then_committed(tmp_path)

    review(lead, old_id, "keep")

    assert [i["id"] for i in to_review(lead, "HEAD~1..HEAD", cwd=str(tmp_path))["to_review"]] == [old_id]


def test_install_hook_writes_a_hook_that_never_blocks_a_commit(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    monkeypatch.setenv("CORTEX_TEAM", "t1")
    monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "missing-dir" / "x.db"))

    hook = install_hook(cwd=str(tmp_path))["installed"]

    text = open(hook).read()
    assert "commit-msg" in text and "CORTEX_TEAM=t1" in text and text.rstrip().endswith("|| true")
    commit(tmp_path, "still commits")
    install_hook(cwd=str(tmp_path))  # reinstalling over our own hook is fine
