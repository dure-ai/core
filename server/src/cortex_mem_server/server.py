"""cortex-mem MCP server (stdio). Turns tool calls into runtime calls:
tool arguments -> Caller + scope policy -> Runtime. Storage, delivery and
checks live in the runtime and the stores; the engine is picked by config.

Environment variables (all optional):
    CORTEX_DB_PATH       SQLite file                    (default: ~/.cortex-mem/cortex.db)
    CORTEX_USER          who is recording               (default: OS user)
    CORTEX_TEAM          the user's team                (default: default)
    CORTEX_AGENT         client name, e.g. claude-code  (default: unset)
    CORTEX_REPO          repository this project works in, e.g. acme/api (default: unset);
                         set it per project (.mcp.json) so scope policy and engines know
                         where work happens without the agent passing repo
    CORTEX_ENGINE        promote_all | no_memory | module:factory   (default: promote_all)
    CORTEX_RECORD_SCOPE  default scope of what is recorded          (default: personal)
    CORTEX_REPO_SCOPES   per-repo defaults, comma-separated fnmatch pattern=scope,
                         e.g. "acme/*=team,acme/handbook=org"
    CORTEX_MAX_FAILURES  failed deliveries of one event before it is skipped (default: 3)

One server process is one session: Claude Code starts a process per session.
Processes share the SQLite file; the runtime delivers each event once.
"""

import getpass
import importlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import get_args
from uuid import uuid4

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from cortex_mem.access import knowledge_visible
from cortex_mem.contract import Caller, Engine, Event, Outcome, Scope, Signal
from cortex_mem.runtime import Runtime
from cortex_mem.store.base import Database
from cortex_mem.store.sqlite import SqliteDatabase

ENGINE_ALIASES = {
    "promote_all": "cortex_mem.engines.baseline:PromoteAllEngine",
    "no_memory": "cortex_mem.engines.baseline:NoMemoryEngine",
}
_WIDTH = {"personal": 0, "team": 1, "org": 2}
log = logging.getLogger("cortex_mem_server")


def _narrow(requested: Scope | None, default: Scope) -> Scope:
    """The requested scope if it is no wider than the default, else the default."""
    if requested is None or _WIDTH[requested] > _WIDTH[default]:
        return default
    return requested


def _scope(value: str) -> Scope:
    if value not in get_args(Scope):
        raise ValueError(f"unknown scope: {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class ScopePolicy:
    """Who sees a record is decided by the deployment, not by the agent: the
    agent may narrow the default but never widen it."""

    default: Scope = "personal"
    repos: tuple[tuple[str, Scope], ...] = ()  # (fnmatch pattern, scope), first match wins

    @classmethod
    def parse(cls, default: str = "personal", repos: str = "") -> "ScopePolicy":
        pairs = []
        for entry in filter(None, (part.strip() for part in repos.split(","))):
            pattern, _, scope = entry.rpartition("=")
            if not pattern:
                raise ValueError(f"expected pattern=scope, got {entry!r}")
            pairs.append((pattern, _scope(scope)))
        return cls(_scope(default), tuple(pairs))

    def default_for(self, repo: str | None) -> Scope:
        if repo:
            for pattern, scope in self.repos:
                if fnmatchcase(repo, pattern):
                    return scope
        return self.default

    def resolve(self, requested: Scope | None, repo: str | None) -> Scope:
        return _narrow(requested, self.default_for(repo))


def load_engine(spec: str) -> tuple[str, Engine]:
    """Resolve an alias or module:factory and build the engine. Returns the
    resolved path too: it names the engine's state, whichever way it was given."""
    path = ENGINE_ALIASES.get(spec, spec)
    module, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"CORTEX_ENGINE must be an alias or module:factory, got {spec!r}")
    factory = getattr(importlib.import_module(module), attr)
    return path, factory({})


class Service:
    """The server's logic without the MCP transport."""

    def __init__(
        self,
        db: Database,
        engine: str,
        user: str,
        team: str,
        session: str,
        agent: str | None = None,
        repo: str | None = None,
        policy: ScopePolicy = ScopePolicy(),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_failures: int | None = None,
    ) -> None:
        name, built = load_engine(engine)
        self.runtime = Runtime(
            built, db.events, db.knowledge, db.state(name), transaction=db.transaction, max_failures=max_failures
        )
        self.user, self.team, self.session, self.agent = user, team, session, agent
        self.repo = repo  # the configured repo: picks the scope policy; a tool argument only says where
        self.policy = policy
        self.clock = clock
        self._skipped_logged = 0

    def _caller(self, repo: str | None = None, context: Mapping[str, str] | None = None) -> Caller:
        caller = Caller(
            user=self.user, team=self.team, session=self.session, at=self.clock(), agent=self.agent,
            repo=repo or self.repo, context=context or {},
        )
        try:
            self.runtime.tick(caller.at)  # delivers what other sessions appended, then lets time pass
        except Exception:
            # An engine failure must not stop the tool call; the runtime retries
            # the event next time and skips it after max_failures attempts.
            log.exception("engine failed while catching up; continuing")
        self._log_skipped()
        return caller

    def _append(self, caller: Caller, **fields) -> tuple[str, bool]:
        """Append an event; returns its id and whether the engine has seen it.
        Once stored, the event is not lost if delivery fails: it is retried
        later, so the agent must not record it again."""
        event = Event(id=uuid4().hex, at=caller.at, user=caller.user, team=caller.team,
                      session=caller.session, agent=caller.agent, **fields)
        try:
            self.runtime.append(event)
        except Exception:
            if not self.runtime.events.get([event.id]):
                raise  # refused before it was stored (e.g. missing refs)
            log.exception("event %s stored, but the engine failed to take it in", event.id)
            return event.id, False
        finally:
            self._log_skipped()
        return event.id, True

    def _log_skipped(self) -> None:
        for skipped in self.runtime.skipped[self._skipped_logged:]:
            log.error(
                "skipped event %s after %d failed deliveries: %s",
                skipped.event_id, skipped.attempts, skipped.error,
            )
        self._skipped_logged = len(self.runtime.skipped)

    def record(
        self,
        text: str,
        scope: Scope | None = None,
        conditions: dict[str, str] | None = None,
        refs: Sequence[str] = (),
        repo: str | None = None,
        rev: str | None = None,
    ) -> dict:
        caller = self._caller(repo)
        effective = self.policy.resolve(scope, self.repo)
        id, delivered = self._append(
            caller, kind="record", scope=effective, text=text, signal="self",
            conditions=conditions or {}, refs=tuple(refs), repo=caller.repo, rev=rev,
        )
        return {"id": id, "scope": effective, "delivered": delivered}

    def recall(
        self,
        query: str,
        scope: Scope = "org",
        k: int = 6,
        repo: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> list[dict]:
        caller = self._caller(repo, context)
        handoffs = self.runtime.recall(query, caller, scope, event_id=uuid4().hex, k=k)
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
        scope: Scope | None = None,
        repo: str | None = None,
    ) -> dict:
        caller = self._caller(repo)
        effective = _narrow(scope, self._outcome_default(caller, refs))
        id, delivered = self._append(
            caller, kind="outcome", scope=effective, text=text, outcome=outcome,
            signal=signal, refs=tuple(refs), repo=caller.repo,
        )
        return {"id": id, "scope": effective, "delivered": delivered}

    def _outcome_default(self, caller: Caller, refs: Sequence[str]) -> Scope:
        """An outcome is evidence about what it judges, so by default it is
        seen by the widest audience of the judged knowledge the caller can see.
        A team item can only be demoted by evidence the team sees."""
        judged = [k for k in self.runtime.knowledge.get(refs) if knowledge_visible(k, caller, "org")]
        if not judged:
            return self.policy.default_for(self.repo)
        return max((k.scope for k in judged), key=_WIDTH.__getitem__)


mcp = MCPServer("cortex-mem")
_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        path = Path(os.environ.get("CORTEX_DB_PATH", "~/.cortex-mem/cortex.db")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        _service = Service(
            SqliteDatabase(path),
            engine=os.environ.get("CORTEX_ENGINE", "promote_all"),
            user=os.environ.get("CORTEX_USER") or getpass.getuser(),
            team=os.environ.get("CORTEX_TEAM", "default"),
            session=uuid4().hex,
            agent=os.environ.get("CORTEX_AGENT"),
            repo=os.environ.get("CORTEX_REPO") or None,
            policy=ScopePolicy.parse(
                os.environ.get("CORTEX_RECORD_SCOPE", "personal"), os.environ.get("CORTEX_REPO_SCOPES", "")
            ),
            max_failures=int(os.environ.get("CORTEX_MAX_FAILURES", "3")),
        )
    return _service


@mcp.tool()
def record_episode(
    text: str,
    scope: Scope | None = None,
    conditions: dict[str, str] | None = None,
    refs: list[str] | None = None,
    repo: str | None = None,
    rev: str | None = None,
) -> dict:
    """Record something learned while working (a fix, a cause, a convention) so
    future sessions can recall it. Who may see it is set by the server for the
    repo; scope can only narrow that (e.g. "personal" for a private note).
    conditions: when it holds, e.g. {"service": "api"}. refs: ids of recalled
    memories this builds on. repo/rev: where the evidence is (repo, commit);
    repo defaults to the project's configured repo and does not change who may
    see the record. Returns the id, the scope actually used, and delivered:
    false means it is stored and will reach memory later, so do not record it
    again."""
    return get_service().record(text, scope, conditions, refs or (), repo, rev)


@mcp.tool()
def recall(
    query: str,
    scope: Scope = "org",
    k: int = 6,
    repo: str | None = None,
    context: dict[str, str] | None = None,
) -> dict:
    """Recall memories relevant to the current task. Call before starting work
    that past sessions may have covered. By default searches everything you may
    see (your own, your team's, the org's); scope narrows it. repo and context
    say where you are working (e.g. context={"service": "api"}), so facts that
    hold only elsewhere can be left out; repo defaults to the project's
    configured repo. Returns up to k memories, best first."""
    return {"memories": get_service().recall(query, scope, k, repo, context)}


@mcp.tool()
def report_outcome(
    outcome: Outcome,
    refs: list[str],
    text: str = "",
    signal: Signal = "self",
    scope: Scope | None = None,
    repo: str | None = None,
) -> dict:
    """Report whether recalled memories (refs = their ids, at least one) helped:
    "success", "failure", or "unknown". signal says who judged it: "self", "test",
    "review", "human", or "doc". This is how memories earn or lose trust.
    By default the report is visible to whoever can see the judged memories;
    scope can only narrow that. Returns the id, the scope actually used, and
    delivered (false: stored, reaches memory later; do not report it again)."""
    try:
        return get_service().report_outcome(outcome, refs, text, signal, scope, repo)
    except ValueError as exc:  # e.g. empty refs: tell the agent why, not just that it failed
        raise ToolError(str(exc)) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    mcp.run()


if __name__ == "__main__":
    main()
