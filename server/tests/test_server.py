import json
import os
import sys
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

import anyio
import pytest
from cortex_mem_server import server as srv
from cortex_mem_server.server import ScopePolicy, Service, load_engine
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cortex_mem.contract import Demote
from cortex_mem.embed import UpstageEmbedder
from cortex_mem.index import Index
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

    assert good["delivered"] is True and good["pending"] == 0
    assert [h["content"] for h in m.recall("deploy tags")] == ["deploy uses release tags"]
    assert [s.event_id for s in m.runtime.skipped] == [bad["id"]]
    assert f"skipped event {bad['id']} after 3 failed deliveries" in caplog.text


def test_without_max_failures_the_service_keeps_retrying():
    db = SqliteDatabase()
    m = Service(db, f"{__name__}:FailsOnBoom", user="u1", team="t1", session="s1", clock=lambda: T0)

    m.record("boom")
    later = m.record("deploy uses release tags")

    assert later["delivered"] is False and m.runtime.skipped == []
    assert later["pending"] == 2  # both records wait behind the one the engine fails on
    assert m.runtime.state.cursor() == 0


def test_record_during_an_embedding_outage_is_recorded_but_pending():
    from cortex_mem.embed import EmbeddingError, HashEmbedder

    class Flaky(HashEmbedder):
        down = False

        def embed(self, texts, kind):
            if self.down:
                raise EmbeddingError("embedding backend is down")
            return super().embed(texts, kind)

    db = SqliteDatabase()
    emb = Flaky()
    m = Service(db, "promote_all", user="u1", team="t1", session="s1", clock=lambda: T0, max_failures=3,
                index=Index(db.vectors, emb, 0.9, transaction=db.transaction))
    emb.down = True

    stored = m.record("deploy uses release tags")

    assert stored["delivered"] is False and stored["pending"] >= 1
    assert db.events.get([stored["id"]])  # recorded, not lost
    assert m.runtime.skipped == []

    emb.down = False
    later = m.record("payments need vendor v2")
    assert later["delivered"] is True and later["pending"] == 0
    assert m.runtime.state.cursor() == 2 and m.runtime.skipped == []


def test_delivered_responses_say_how_many_events_wait():
    m = service(SqliteDatabase())
    recorded = m.record("x")
    assert recorded == {"id": recorded["id"], "scope": "personal", "delivered": True, "pending": 0}
    out = m.report_outcome("success", [recorded["id"]])
    assert out["delivered"] is True and out["pending"] == 0


def test_a_failing_pending_count_still_reports_the_event_as_stored(caplog):
    m = service(SqliteDatabase())

    def down():
        raise OSError("store is down")

    m.runtime.pending = down
    out = m.record("deploy uses release tags")

    assert out["delivered"] is False and out["pending"] is None
    assert m.db.events.get([out["id"]])  # stored: the agent must not record it again
    assert "deploy uses release tags" not in caplog.text


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


# ── embeddings config ───────────────────────────────────────────────────────

_EMBED_VARS = ("CORTEX_EMBEDDER", "CORTEX_EMBED_PASSAGE_MODEL", "CORTEX_EMBED_QUERY_MODEL", "UPSTAGE_API_KEY",
               "CORTEX_GROUP_THRESHOLD", "CORTEX_TEAM", "CORTEX_USER",
               "CORTEX_DB_URL", "CORTEX_DB_LOCK_TIMEOUT", "CORTEX_VECTOR_INDEX")


def _service_from_env(monkeypatch, tmp_path, **env):
    for name in _EMBED_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_TEAM", "t1")
    monkeypatch.setenv("CORTEX_USER", "u1")
    monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "cortex.db"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(srv, "_service", None)
    return srv.get_service()


def test_embedder_none_by_default_runs_without_an_index(monkeypatch, tmp_path):
    svc = _service_from_env(monkeypatch, tmp_path)

    assert svc.runtime.index is None
    assert srv._embedder() is None


def test_hash_embedder_builds_an_index(monkeypatch, tmp_path):
    svc = _service_from_env(monkeypatch, tmp_path, CORTEX_EMBEDDER=" Hash ")
    first = svc.record("deploy uses release tags")["id"]
    svc.record("deploy uses release tags")

    index = svc.runtime.index
    assert isinstance(index, Index)
    group = index.reader(svc.runtime.events, svc.runtime.knowledge).group(first)
    assert group is not None and len(group.members) == 2


def test_upstage_needs_models_and_key(monkeypatch):
    for name in _EMBED_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_EMBEDDER", "upstage")
    monkeypatch.setenv("CORTEX_EMBED_PASSAGE_MODEL", "passage-m")
    with pytest.raises(SystemExit) as exc:
        srv._embedder()
    message = str(exc.value)
    assert "CORTEX_EMBED_QUERY_MODEL" in message and "UPSTAGE_API_KEY" in message
    assert "CORTEX_EMBED_PASSAGE_MODEL" not in message

    monkeypatch.setenv("CORTEX_EMBED_QUERY_MODEL", "query-m")
    monkeypatch.setenv("UPSTAGE_API_KEY", "   ")
    with pytest.raises(SystemExit, match="UPSTAGE_API_KEY"):
        srv._embedder()

    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-secret-value")
    built = srv._embedder()  # builds the client; makes no request
    assert isinstance(built, UpstageEmbedder)
    assert "sk-secret-value" not in repr(built)


def test_upstage_error_never_prints_the_key(monkeypatch):
    monkeypatch.setenv("CORTEX_EMBEDDER", "upstage")
    monkeypatch.delenv("CORTEX_EMBED_PASSAGE_MODEL", raising=False)
    monkeypatch.delenv("CORTEX_EMBED_QUERY_MODEL", raising=False)
    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-secret-value")
    with pytest.raises(SystemExit) as exc:
        srv._embedder()
    assert "sk-secret-value" not in str(exc.value)


@pytest.mark.parametrize(
    "name, value",
    [("CORTEX_EMBEDDER", "foo"), ("CORTEX_GROUP_THRESHOLD", "1.5"), ("CORTEX_GROUP_THRESHOLD", "abc"),
     ("CORTEX_GROUP_THRESHOLD", "-0.1"), ("CORTEX_GROUP_THRESHOLD", "nan")],
)
def test_bad_embedder_name_and_threshold_stop_the_server(monkeypatch, name, value):
    for var in _EMBED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(name, value)
    check = srv._embedder if name == "CORTEX_EMBEDDER" else srv._group_threshold
    with pytest.raises(SystemExit) as exc:
        check()
    assert value in str(exc.value)


def test_group_threshold_defaults_to_point_nine(monkeypatch):
    monkeypatch.delenv("CORTEX_GROUP_THRESHOLD", raising=False)
    assert srv._group_threshold() == 0.9
    monkeypatch.setenv("CORTEX_GROUP_THRESHOLD", "0")
    assert srv._group_threshold() == 0.0


@pytest.mark.parametrize("key", ["sk-secret\r\nX: 1", "sk-비밀-key"])
def test_upstage_key_that_cannot_be_sent_stops_the_server_without_printing_it(monkeypatch, key):
    monkeypatch.setenv("CORTEX_EMBEDDER", "upstage")
    monkeypatch.setenv("CORTEX_EMBED_PASSAGE_MODEL", "passage-m")
    monkeypatch.setenv("CORTEX_EMBED_QUERY_MODEL", "query-m")
    monkeypatch.setenv("UPSTAGE_API_KEY", key)
    with pytest.raises(SystemExit) as exc:
        srv._embedder()
    assert "UPSTAGE_API_KEY" in str(exc.value)
    assert "sk-secret" not in str(exc.value) and "비밀" not in str(exc.value)


def test_upstage_timeout_is_shorter_than_sqlites_busy_timeout(monkeypatch):
    from cortex_mem.store.sqlite import BUSY_TIMEOUT_MS

    monkeypatch.setenv("CORTEX_EMBEDDER", "upstage")
    monkeypatch.setenv("CORTEX_EMBED_PASSAGE_MODEL", "passage-m")
    monkeypatch.setenv("CORTEX_EMBED_QUERY_MODEL", "query-m")
    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-secret-value")
    # A promoted item is embedded while the write lock is held (spec §7).
    assert srv._embedder().timeout < BUSY_TIMEOUT_MS / 1000


# ── PostgreSQL config ───────────────────────────────────────────────────────

_DB_VARS = ("CORTEX_DB_URL", "CORTEX_DB_LOCK_TIMEOUT", "CORTEX_VECTOR_INDEX")


def test_sqlite_is_the_default_and_the_url_is_unset(monkeypatch, tmp_path):
    for name in _DB_VARS:
        monkeypatch.delenv(name, raising=False)
    svc = _service_from_env(monkeypatch, tmp_path)

    assert srv._db_url() is None
    assert srv._lock_timeout() == "30s" and srv._vector_index() == "none"
    assert (tmp_path / "cortex.db").exists()
    assert svc.runtime.events is not None


def test_blank_url_counts_as_unset(monkeypatch):
    monkeypatch.setenv("CORTEX_DB_URL", "   ")
    assert srv._db_url() is None


@pytest.mark.parametrize(
    "name, value, check",
    [("CORTEX_DB_LOCK_TIMEOUT", "soon", "_lock_timeout"), ("CORTEX_DB_LOCK_TIMEOUT", "", "_lock_timeout"),
     ("CORTEX_VECTOR_INDEX", "ivfflat", "_vector_index")],
)
def test_bad_lock_timeout_and_vector_index_stop_the_server(monkeypatch, name, value, check):
    for var in _DB_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        getattr(srv, check)()


def test_lock_timeout_and_vector_index_accept_good_values(monkeypatch):
    monkeypatch.setenv("CORTEX_DB_LOCK_TIMEOUT", " 500ms ")
    monkeypatch.setenv("CORTEX_VECTOR_INDEX", " HNSW ")
    assert srv._lock_timeout() == "500ms" and srv._vector_index() == "hnsw"


def test_connection_refused_exits_without_the_password(monkeypatch, tmp_path, caplog):
    url = "postgresql://postgres:wrong-secret@127.0.0.1:1/postgres"
    with pytest.raises(SystemExit) as exc:
        _service_from_env(monkeypatch, tmp_path, CORTEX_DB_URL=url)
    assert "could not open the PostgreSQL store" in str(exc.value)
    assert "wrong-secret" not in str(exc.value) and "wrong-secret" not in caplog.text


def test_wrong_password_exits_without_the_password(monkeypatch, tmp_path, caplog):
    import pg

    base = pg.pg_url()
    if base is None:
        pytest.skip("CORTEX_TEST_PG_URL is not set")
    parts = urlsplit(base)
    url = urlunsplit(parts._replace(netloc=f"{parts.username or 'postgres'}:wrong-secret@{parts.hostname}:{parts.port}", query=""))
    with pytest.raises(SystemExit) as exc:
        _service_from_env(monkeypatch, tmp_path, CORTEX_DB_URL=url)
    assert "could not open the PostgreSQL store" in str(exc.value)
    assert "wrong-secret" not in str(exc.value) and "wrong-secret" not in caplog.text


def test_url_wins_over_the_sqlite_path_with_a_warning(monkeypatch, tmp_path, pg_schema, caplog):
    caplog.set_level("WARNING", logger="cortex_mem_server")
    svc = _service_from_env(monkeypatch, tmp_path, CORTEX_DB_URL=pg_schema)
    try:
        assert svc.db.kind == "postgres"
        svc.record("deploy uses release tags")
        assert not (tmp_path / "cortex.db").exists()
        assert "CORTEX_DB_PATH" in caplog.text and pg_schema not in caplog.text
    finally:
        svc.db.close()


def test_postgres_service_records_events_and_groups(monkeypatch, tmp_path, pg_schema):
    svc = _service_from_env(
        monkeypatch, tmp_path, CORTEX_DB_URL=pg_schema, CORTEX_EMBEDDER="hash", CORTEX_VECTOR_INDEX="none"
    )
    try:
        first = svc.record("deploy uses release tags")["id"]
        svc.record("deploy uses release tags")
        group = svc.runtime.index.reader(svc.runtime.events, svc.runtime.knowledge).group(first)
        assert group is not None and len(group.members) == 2
    finally:
        svc.db.close()
