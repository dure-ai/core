import json
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cortex_mem.engines.baseline import PromoteAllEngine
from cortex_mem.server import Memory
from cortex_mem.store.sqlite import SqliteEpisodeStore


def memory(store, user="u1", session="s1"):
    return Memory(store, PromoteAllEngine, user=user, session=session)


def test_recall_returns_recorded_episode():
    store = SqliteEpisodeStore()
    m = memory(store)
    rid = m.record("deploy uses release tags")

    assert [h["id"] for h in m.recall("deploy tags")] == [rid]


def test_recall_is_appended_as_recall_event_with_refs():
    store = SqliteEpisodeStore()
    m = memory(store)
    rid = m.record("deploy uses release tags")

    m.recall("deploy tags")

    recall = [e for e in store.events() if e.kind == "recall"]
    assert [(e.text, e.refs) for e in recall] == [("deploy tags", (rid,))]


def test_report_outcome_refs_judged_ids():
    store = SqliteEpisodeStore()
    m = memory(store)
    rid = m.record("deploy uses release tags")

    oid = m.report_outcome("failure", [rid], text="tag missing", signal="test")

    [event] = store.get([oid])
    assert (event.kind, event.outcome, event.signal, event.refs) == ("outcome", "failure", "test", (rid,))


def test_new_session_replays_store_into_engine():
    store = SqliteEpisodeStore()
    rid = memory(store, session="a").record("deploy uses release tags")

    b = memory(store, session="b")

    assert [h.id for h in b.engine.snapshot(store.get([rid])[0].at)] == [rid]


def test_personal_episode_is_hidden_from_other_users_team_recall():
    store = SqliteEpisodeStore()
    memory(store, user="u1").record("deploy uses release tags", scope="personal")
    tid = memory(store, user="u1").record("deploy needs approval", scope="team")

    assert [h["id"] for h in memory(store, user="u2").recall("deploy", scope="team")] == [tid]


def call(db, tool, args):
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "cortex_mem.server"],
            env={"CORTEX_DB_PATH": str(db), "CORTEX_USER": "u1"},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return json.loads(result.content[0].text)

    return anyio.run(run)


def test_stdio_session_b_recalls_what_session_a_recorded(tmp_path):
    db = tmp_path / "episodes.db"
    rid = call(db, "record_episode", {"text": "cloud sql needs the auth proxy"})["id"]

    hits = call(db, "recall", {"query": "cloud sql proxy"})["memories"]

    assert [h["id"] for h in hits] == [rid]
