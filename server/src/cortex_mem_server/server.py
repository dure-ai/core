"""cortex-mem MCP server (stdio). Turns tool calls into runtime calls:
tool arguments -> Caller + scope policy -> Runtime. Storage, delivery and
checks live in the runtime and the stores; the engine is picked by config.

Environment variables (only CORTEX_TEAM is required):
    CORTEX_TEAM          the user's team; the server refuses to start without it, since
                         records filed under the wrong team cannot be moved later
    CORTEX_DB_URL        postgresql://... : use PostgreSQL; its password is never printed
                         (default: unset, SQLite)
    CORTEX_DB_PATH       SQLite file, only without CORTEX_DB_URL (default: ~/.cortex-mem/cortex.db)
    CORTEX_DB_LOCK_TIMEOUT  how long PostgreSQL waits for the write lock, e.g. 30s, 500ms (default: 30s)
    CORTEX_VECTOR_INDEX  none | hnsw: PostgreSQL vector search, exact or approximate (default: none)
    CORTEX_USER          who is recording               (default: OS user, with a notice)
    CORTEX_AGENT         client name, e.g. claude-code  (default: unset)
    CORTEX_REPO          repository this project works in, e.g. acme/api (default: unset);
                         register the server per project (`claude mcp add -s local -e
                         CORTEX_REPO=...`) so it picks the repo's scope policy and tells
                         engines where work happens; the agent cannot change the policy
    CORTEX_ENGINE        promote_all | no_memory | module:factory   (default: promote_all)
    CORTEX_RECORD_SCOPE  default scope of what is recorded          (default: personal)
    CORTEX_REPO_SCOPES   per-repo defaults, comma-separated fnmatch pattern=scope,
                         e.g. "acme/*=team,acme/handbook=org"
    CORTEX_MAX_FAILURES  failed deliveries of one event before it is skipped (default: 3)
    CORTEX_EMBEDDER      none | hash | upstage: embeddings for grouping and similar recall (default: none)
    CORTEX_EMBED_PASSAGE_MODEL, CORTEX_EMBED_QUERY_MODEL
                         Upstage model names; required when CORTEX_EMBEDDER=upstage
    UPSTAGE_API_KEY      Upstage key; required when CORTEX_EMBEDDER=upstage; never printed
    CORTEX_GROUP_THRESHOLD  similarity from 0 to 1 at which records are grouped (default: 0.9)

One server process is one session: Claude Code starts a process per session.
Processes share the SQLite file or the PostgreSQL database; the runtime delivers each event once.
"""

import getpass
import importlib
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, cast, get_args
from urllib.parse import urlsplit
from uuid import uuid4

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from cortex_mem.access import knowledge_visible
from cortex_mem.contract import Caller, Engine, Event, Outcome, Scope, Signal
from cortex_mem.embed import Embedder, HashEmbedder, UpstageEmbedder
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.base import Database
from cortex_mem.store.postgres import PostgresDatabase, check_lock_timeout
from cortex_mem.store.sqlite import SqliteDatabase

ENGINE_ALIASES = {
    "promote_all": "cortex_mem.engines.baseline:PromoteAllEngine",
    "no_memory": "cortex_mem.engines.baseline:NoMemoryEngine",
}
_WIDTH = {"personal": 0, "team": 1, "org": 2}
log = logging.getLogger("cortex_mem_server")
# Nothing is embedded while a write transaction is open: passages, promoted
# items and an engine's query text (ctx.similar.search or .knowledge during
# observe/tick) are all embedded with no write lock held, so a slow or failing
# Upstage call never blocks another session's write. This bound is how long a
# tool call waits on one stalled request (per socket operation, so a response
# that trickles in can take longer) before the delivery fails with an
# EmbeddingError and is retried on the next call.
UPSTAGE_TIMEOUT_S = 4.0


def _narrow(requested: Scope | None, default: Scope) -> Scope:
    """The requested scope if it is no wider than the default, else the default."""
    if requested is None or _WIDTH[requested] > _WIDTH[default]:
        return default
    return requested


def _scope(value: str) -> Scope:
    if value not in get_args(Scope):
        raise ValueError(f"unknown scope: {value!r}")
    return cast(Scope, value)


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
        index: Index | None = None,
    ) -> None:
        name, built = load_engine(engine)
        self.db = db  # exposed so callers and tests can close the store
        self.runtime = Runtime(
            built, db.events, db.knowledge, db.state(name), transaction=db.transaction, max_failures=max_failures,
            index=index,
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

    def _append(self, caller: Caller, **fields) -> tuple[str, bool, int | None]:
        """Append an event; returns its id, whether the engine has taken this
        event in (false while it waits: behind an earlier event, an embedding
        outage or a failing engine; and false when it was skipped after
        max_failures failed deliveries) and how many events still wait, this
        one or others (None when that could not be read).
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
        finally:
            self._log_skipped()
        try:
            waiting = self.runtime.waiting()
        except Exception:
            # The event is stored: answer delivered false, never an error the
            # agent would answer by recording it again.
            log.exception("event %s stored, but its delivery could not be checked", event.id)
            return event.id, False, None
        skipped = any(s.event_id == event.id for s in self.runtime.skipped)
        return event.id, event.id not in waiting and not skipped, len(waiting)

    def pending(self) -> int | None:
        """How many events wait to be delivered, from any session; None when
        that could not be read."""
        try:
            return self.runtime.pending()
        except Exception:
            log.exception("could not count the waiting events")
            return None

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
    ) -> dict[str, Any]:
        caller = self._caller(repo)
        effective = self.policy.resolve(scope, self.repo)
        id, delivered, pending = self._append(
            caller, kind="record", scope=effective, text=text, signal="self",
            conditions=conditions or {}, refs=tuple(refs), repo=caller.repo, rev=rev,
        )
        return {"id": id, "scope": effective, "delivered": delivered, "pending": pending}

    def recall(
        self,
        query: str,
        scope: Scope = "org",
        k: int = 6,
        repo: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
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
        scope: Scope | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        caller = self._caller(repo)
        effective = _narrow(scope, self._outcome_default(caller, refs))
        # The judgement must reach whoever sees the judged item, but the agent's
        # words follow the repo's policy: past it, only the judgement is shared.
        redacted = bool(text) and _WIDTH[effective] > _WIDTH[self.policy.default_for(self.repo)]
        id, delivered, pending = self._append(
            caller, kind="outcome", scope=effective, text="" if redacted else text, outcome=outcome,
            # An agent's report is its own judgement, like its records: a test,
            # review or person must report through another path to count as one.
            signal="self", refs=tuple(refs), repo=caller.repo, text_redacted=redacted,
        )
        return {
            "id": id, "scope": effective, "signal": "self", "text_redacted": redacted, "delivered": delivered,
            "pending": pending,
        }

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


def _identity() -> tuple[str, str]:
    """(user, team) from the environment. A missing team stops the server."""
    team = os.environ.get("CORTEX_TEAM", "").strip()
    if not team:
        raise SystemExit(
            "cortex-mem: set CORTEX_TEAM to your team (e.g. -e CORTEX_TEAM=payments); "
            "team records are shared by this name and cannot be moved later"
        )
    user = os.environ.get("CORTEX_USER", "").strip()
    if not user:
        user = getpass.getuser()
        log.warning("CORTEX_USER is not set; recording as the OS user %r", user)
    return user, team


def _max_failures() -> int:
    raw = os.environ.get("CORTEX_MAX_FAILURES", "3")
    if not raw.strip().isdigit() or int(raw) < 1:
        raise SystemExit(f"cortex-mem: CORTEX_MAX_FAILURES must be a whole number of at least 1, got {raw!r}")
    return int(raw)


def _embedder() -> Embedder | None:
    """The embedder CORTEX_EMBEDDER names: none (the default: no embeddings),
    hash, or upstage, which also needs CORTEX_EMBED_PASSAGE_MODEL,
    CORTEX_EMBED_QUERY_MODEL and UPSTAGE_API_KEY. Exits with a message that
    never holds the key when a setting is missing or invalid."""
    name = os.environ.get("CORTEX_EMBEDDER", "none").strip().lower()
    if name == "none":
        return None
    if name == "hash":
        return HashEmbedder()
    if name == "upstage":
        passage = os.environ.get("CORTEX_EMBED_PASSAGE_MODEL", "").strip()
        query = os.environ.get("CORTEX_EMBED_QUERY_MODEL", "").strip()
        key = os.environ.get("UPSTAGE_API_KEY", "").strip()
        missing = [
            var
            for var, value in (
                ("CORTEX_EMBED_PASSAGE_MODEL", passage), ("CORTEX_EMBED_QUERY_MODEL", query), ("UPSTAGE_API_KEY", key)
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"cortex-mem: CORTEX_EMBEDDER=upstage also needs {', '.join(missing)}")
        try:
            return UpstageEmbedder(key, passage, query, timeout=UPSTAGE_TIMEOUT_S)
        except ValueError:
            pass  # exit after the handler, so nothing is chained; the message names no value

        raise SystemExit("cortex-mem: UPSTAGE_API_KEY must be printable ASCII without control characters")
    raise SystemExit(f"cortex-mem: CORTEX_EMBEDDER must be none, hash or upstage, got {name!r}")


def _group_threshold() -> float:
    """CORTEX_GROUP_THRESHOLD, a number from 0 to 1 (default 0.9); exits
    otherwise."""
    raw = os.environ.get("CORTEX_GROUP_THRESHOLD", "0.9")
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:  # false for nan too
        raise SystemExit(f"cortex-mem: CORTEX_GROUP_THRESHOLD must be a number from 0 to 1, got {raw!r}")
    return value


def _db_url() -> str | None:
    """CORTEX_DB_URL, or None when it is unset or blank (then SQLite is used)."""
    return os.environ.get("CORTEX_DB_URL", "").strip() or None


def _lock_timeout() -> str:
    """CORTEX_DB_LOCK_TIMEOUT as a PostgreSQL duration (default 30s); exits
    otherwise."""
    raw = os.environ.get("CORTEX_DB_LOCK_TIMEOUT", "30s").strip()
    try:
        return check_lock_timeout(raw)
    except ValueError as exc:
        raise SystemExit(f"cortex-mem: {exc}") from None


def _vector_index() -> str:
    """CORTEX_VECTOR_INDEX: none (the default, exact search) or hnsw; exits
    otherwise."""
    name = os.environ.get("CORTEX_VECTOR_INDEX", "none").strip().lower()
    if name not in ("none", "hnsw"):
        raise SystemExit(f"cortex-mem: CORTEX_VECTOR_INDEX must be none or hnsw, got {name!r}")
    return name


def _open_database() -> Database:
    """PostgreSQL when CORTEX_DB_URL is set, else SQLite at CORTEX_DB_PATH
    (default ~/.cortex-mem/cortex.db; its directory is created). Exits with
    a message that holds neither the URL nor the password when PostgreSQL
    cannot be reached."""
    url = _db_url()
    if url is None:
        path = Path(os.environ.get("CORTEX_DB_PATH", "~/.cortex-mem/cortex.db")).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDatabase(path)
    if os.environ.get("CORTEX_DB_PATH"):
        log.warning("CORTEX_DB_URL and CORTEX_DB_PATH are both set; using PostgreSQL and ignoring the SQLite path")
    try:
        return PostgresDatabase(url, _lock_timeout(), _vector_index())
    except ConnectionError as exc:
        # The error carries host, port and database, and the password is redacted already;
        # do it again here in case the driver quoted the URL.
        password = urlsplit(url).password
        reason = str(exc).replace(url, "<url>")
        if password:
            reason = reason.replace(password, "***")
        raise SystemExit(f"cortex-mem: could not open the PostgreSQL store: {reason} ({type(exc).__name__})") from None


def get_service() -> Service:
    global _service
    if _service is None:
        user, team = _identity()
        db = _open_database()
        embedder, threshold = _embedder(), _group_threshold()
        _service = Service(
            db,
            engine=os.environ.get("CORTEX_ENGINE", "promote_all"),
            user=user,
            team=team,
            session=uuid4().hex,
            agent=os.environ.get("CORTEX_AGENT"),
            repo=os.environ.get("CORTEX_REPO") or None,
            policy=ScopePolicy.parse(
                os.environ.get("CORTEX_RECORD_SCOPE", "personal"), os.environ.get("CORTEX_REPO_SCOPES", "")
            ),
            max_failures=_max_failures(),
            index=None if embedder is None else Index(db.vectors, embedder, threshold, transaction=db.transaction),
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
) -> dict[str, Any]:
    """Record something learned while working (a fix, a cause, a convention) so
    future sessions can recall it. Who may see it is set by the server for the
    repo; scope can only narrow that (e.g. "personal" for a private note).
    conditions: when it holds, e.g. {"service": "api"}. refs: ids of recalled
    memories this builds on. repo/rev: where the evidence is (repo, commit);
    repo defaults to the project's configured repo and does not change who may
    see the record. Returns the id, the scope actually used, delivered and
    pending: delivered false means it is stored and will reach memory later
    (it is among the pending events that wait), so do not record it again;
    rarely, it was skipped after repeated engine failures, which the server
    logs.
    pending counts every event that waits, from any session, as an integer,
    or null when it could not be read (delivered is then false; the record is
    still stored)."""
    return get_service().record(text, scope, conditions, refs or (), repo, rev)


@mcp.tool()
def recall(
    query: str,
    scope: Scope = "org",
    k: int = 6,
    repo: str | None = None,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Recall memories relevant to the current task. Call before starting work
    that past sessions may have covered. By default searches everything you may
    see (your own, your team's, the org's); scope narrows it. repo and context
    say where you are working (e.g. context={"service": "api"}), so facts that
    hold only elsewhere can be left out; repo defaults to the project's
    configured repo. Returns up to k memories, best first, and pending: how
    many recorded events have not reached memory yet (an integer, or null when
    it could not be read). While pending is above 0, the answer may miss them,
    for example during an embedding outage; they arrive on later calls."""
    service = get_service()
    return {"memories": service.recall(query, scope, k, repo, context), "pending": service.pending()}


@mcp.tool()
def report_outcome(
    outcome: Outcome,
    refs: list[str],
    text: str = "",
    signal: Signal = "self",
    scope: Scope | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    """Report whether recalled memories (refs = their ids, at least one) helped:
    "success", "failure", or "unknown". This is how memories earn or lose trust.
    The report is stored as your own judgement (signal "self") whatever signal
    you pass; test, review and human results count only when they report
    themselves. By default the judgement (outcome, signal, refs) is visible to
    whoever can see the judged memories, e.g. the whole org for an org memory;
    scope can only narrow that. If that is wider than this repo's policy, text
    is not stored (text_redacted: true). Returns the id, the scope and signal
    actually used, text_redacted, delivered and pending (delivered false:
    stored, reaches memory later, pending events wait; do not report it again;
    rarely, skipped after repeated engine failures, which the server logs).
    pending is an integer, or null when it could not be read (delivered is then
    false)."""
    del signal  # kept so clients that send it still work; the server sets the signal
    try:
        return get_service().report_outcome(outcome, refs, text, scope, repo)
    except ValueError as exc:  # e.g. empty refs: tell the agent why, not just that it failed
        raise ToolError(str(exc)) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    _identity()  # fail at start, not at the first tool call
    _max_failures()
    _embedder()
    _group_threshold()
    _lock_timeout()
    _vector_index()
    mcp.run()


if __name__ == "__main__":
    main()
