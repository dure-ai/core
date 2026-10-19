"""cortex-mem MCP server (stdio). Records episodes and hands off memories
through the engine port; every call becomes an event in the store.

Environment variables (all optional):
    CORTEX_DB_PATH  episode store               (default: ~/.cortex-mem/episodes.db)
    CORTEX_USER     who is recording            (default: OS user)
    CORTEX_ENGINE   promote_all | no_memory     (default: promote_all)

One server process is one session: Claude Code starts a process per session.
"""

import getpass
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mcp.server.mcpserver import MCPServer

from .contract import EngineFactory, Event, Kind, Outcome, QueryContext, Scope, Signal
from .engines.baseline import NoMemoryEngine, PromoteAllEngine
from .store.sqlite import SqliteEpisodeStore

ENGINES: dict[str, EngineFactory] = {
    "promote_all": PromoteAllEngine,
    "no_memory": NoMemoryEngine,
}


class Memory:
    """The server's logic without the MCP transport."""

    def __init__(self, store: SqliteEpisodeStore, factory: EngineFactory, user: str, session: str) -> None:
        self.store = store
        self.engine = factory(store)
        self.user = user
        self.session = session
        # Engines are rebuilt from the event sequence, so a new session sees past ones.
        for event in store.events():
            self.engine.observe(event)

    def _emit(self, kind: Kind, scope: Scope, **fields) -> str:
        event = Event(
            id=uuid4().hex, kind=kind, at=datetime.now(UTC),
            user=self.user, session=self.session, scope=scope, **fields,
        )
        self.store.append(event)
        self.engine.observe(event)
        return event.id

    def record(
        self,
        text: str,
        scope: Scope = "personal",
        conditions: dict[str, str] | None = None,
        refs: Sequence[str] = (),
        repo: str | None = None,
        rev: str | None = None,
    ) -> str:
        return self._emit(
            "record", scope, text=text, conditions=conditions or {}, refs=tuple(refs), repo=repo, rev=rev
        )

    def recall(self, query: str, scope: Scope = "personal", k: int = 6) -> list[dict]:
        ctx = QueryContext(user=self.user, session=self.session, scope=scope)
        handoffs = self.engine.handoff(query, ctx, k=k)
        self._emit("recall", scope, text=query, refs=tuple(h.id for h in handoffs))
        return [
            {"id": h.id, "content": h.content, "status": h.status, "conditions": dict(h.conditions)}
            for h in handoffs
        ]

    def report_outcome(
        self,
        outcome: Outcome,
        refs: Sequence[str],
        text: str = "",
        signal: Signal = "self",
        scope: Scope = "personal",
    ) -> str:
        return self._emit("outcome", scope, text=text, outcome=outcome, signal=signal, refs=tuple(refs))


mcp = MCPServer("cortex-mem")
_memory: Memory | None = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        path = Path(os.environ.get("CORTEX_DB_PATH", "~/.cortex-mem/episodes.db")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        _memory = Memory(
            SqliteEpisodeStore(path),
            ENGINES[os.environ.get("CORTEX_ENGINE", "promote_all")],
            user=os.environ.get("CORTEX_USER") or getpass.getuser(),
            session=uuid4().hex,
        )
    return _memory


@mcp.tool()
def record_episode(
    text: str,
    scope: Scope = "personal",
    conditions: dict[str, str] | None = None,
    refs: list[str] | None = None,
    repo: str | None = None,
    rev: str | None = None,
) -> dict:
    """Record something learned while working (a fix, a cause, a convention) so
    future sessions can recall it. scope "team" shares it with teammates.
    conditions: when it holds, e.g. {"service": "api"}. refs: ids of recalled
    memories this builds on. repo/rev: where the evidence is (repo, commit)."""
    return {"id": get_memory().record(text, scope, conditions, refs or (), repo, rev)}


@mcp.tool()
def recall(query: str, scope: Scope = "personal", k: int = 6) -> dict:
    """Recall memories relevant to the current task. Call before starting work
    that past sessions may have covered. scope "team" includes teammates'
    shared memories. Returns up to k memories, best first."""
    return {"memories": get_memory().recall(query, scope, k)}


@mcp.tool()
def report_outcome(
    outcome: Outcome,
    refs: list[str],
    text: str = "",
    signal: Signal = "self",
    scope: Scope = "personal",
) -> dict:
    """Report whether recalled memories (refs = their ids) helped: "success",
    "failure", or "unknown". signal says who judged it: "self", "test",
    "review", or "human". This is how memories earn or lose trust."""
    return {"id": get_memory().report_outcome(outcome, refs, text, signal, scope)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
