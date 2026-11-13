"""cortex-mem MCP server (stdio). Turns tool calls into runtime calls:
tool arguments -> Caller + scope policy -> Runtime. Storage, delivery and
checks live in the runtime and the stores; the engine is picked by config.

Environment variables (only CORTEX_TEAM is required):
    CORTEX_TEAM          the user's team; the server refuses to start without it, since
                         records filed under the wrong team cannot be moved later
    CORTEX_DB_PATH       SQLite file                    (default: ~/.cortex-mem/cortex.db)
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

from cortex_mem.access import event_visible, knowledge_visible
from cortex_mem.contract import Caller, Engine, Event, Outcome, Scope, Signal
from cortex_mem.runtime import Runtime
from cortex_mem.store.base import Database
from cortex_mem.store.sqlite import SqliteDatabase

ENGINE_ALIASES = {
    "promote_all": "cortex_mem.engines.baseline:PromoteAllEngine",
    "no_memory": "cortex_mem.engines.baseline:NoMemoryEngine",
}
_WIDTH = {"personal": 0, "team": 1, "org": 2}
_SERVER_STATE = "cortex_mem_server"  # name of the server's own key-value state in the database
TRAILER = "Cortex-Record"  # commit message trailer naming a record the commit bears out
RECALLED_TRAILER = "Cortex-Recalled"  # ... naming knowledge the committer was handed while making it
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
        self._cited: set[str] = set()  # what this session's records cite as handed (one process, one session)
        self._sessions = db.state(_SERVER_STATE)  # the server's own notes: which project a session runs in
        self._session_noted = False

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
        signal: Signal = "self",
    ) -> dict:
        caller = self._caller(repo)
        self._cited.update(refs)
        effective = self.policy.resolve(scope, self.repo)
        id, delivered = self._append(
            caller, kind="record", scope=effective, text=text, signal=signal,
            conditions=conditions or {}, refs=tuple(refs), repo=caller.repo, rev=rev,
        )
        result: dict = {"id": id, "scope": effective, "delivered": delivered, "commit_trailer": f"{TRAILER}: {id}"}
        if delivered and (related := self._related(text, caller, id)):
            result["related"] = related
            result["next"] = (
                "Existing team knowledge on the same topic is listed in 'related'. If what you just "
                "recorded replaces one of them, call report_outcome(outcome='failure', refs=[its id], "
                f"replaced_by='{id}'). It keeps standing until your record passes a test or a review. "
                "If they are unrelated or still hold, do nothing."
            )
        return result

    def _note_session(self) -> None:
        """Note that this session runs in the configured repo. A recall event's
        repo is where the agent said the question was asked, and agents make
        that up; which project the server runs in is the server's own
        knowledge. recalled() goes by this note. Never fails the tool call."""
        if self._session_noted or not self.repo:
            return
        try:
            at = self._sessions.cursor()
            self._sessions.commit(at, at, {f"session:{self.session}": self.repo}, frozenset())
            self._session_noted = True
        except Exception:
            log.exception("noting the project of session %s failed", self.session)

    def _related(self, text: str, caller: Caller, record_id: str, k: int = 3) -> list[dict]:
        """Knowledge the engine hands over for the record's own text. What is
        handed back is logged as a recall, like any handoff, so the engine
        knows this session has seen it; nothing is logged when nothing is
        found. A failure here never fails the record."""
        try:
            handoffs = [
                h for h in self.runtime.handoff(text, caller, "org", k + 1)
                if h.status == "knowledge" and record_id not in h.sources
            ][:k]
            if handoffs:
                self._note_session()
                self._append(
                    caller, kind="recall", scope="personal", text=text, refs=tuple(h.id for h in handoffs),
                    repo=caller.repo, conditions=caller.context,
                )
        except Exception:
            log.exception("looking up knowledge related to record %s failed", record_id)
            return []
        return [{"id": h.id, "content": h.content, "conditions": dict(h.conditions)} for h in handoffs]

    def recall(
        self,
        query: str,
        scope: Scope = "org",
        k: int = 6,
        repo: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> list[dict]:
        caller = self._caller(repo, context)
        self._note_session()
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
        replaced_by: str | None = None,
        rev: str | None = None,
    ) -> dict:
        caller = self._caller(repo)
        effective = _narrow(scope, self._outcome_default(caller, refs))
        restated = self._restate(caller, replaced_by, outcome, refs) if replaced_by else None
        # The judgement must reach whoever sees the judged item, but the agent's
        # words follow the repo's policy: past it, only the judgement is shared.
        redacted = bool(text) and _WIDTH[effective] > _WIDTH[self.policy.default_for(self.repo)]
        id, delivered = self._append(
            caller, kind="outcome", scope=effective, text="" if redacted else text, outcome=outcome,
            signal=signal, refs=tuple(refs), repo=caller.repo, rev=rev, text_redacted=redacted,
        )
        result = {"id": id, "scope": effective, "text_redacted": redacted, "delivered": delivered}
        if restated:
            result["restated"] = restated
            self._cited.update(refs)
        elif outcome == "failure" and not self._cited.intersection(refs):
            result["next"] = (
                "You reported these memories as not holding, but memory does not yet say what holds instead: "
                "the text of a report is not handed to anyone. If you know what is right, record it with "
                f"record_episode(text=..., refs={list(refs)!r}). Recorded that way it stands in for them once "
                "it passes an outside check. If you only know that they are wrong, do nothing."
            )
        return result

    def _restate(self, caller: Caller, record_id: str, outcome: Outcome, refs: Sequence[str]) -> str:
        """Say again, in this session, the caller's own record, this time citing
        what it replaces. Events are never edited, so the tie is a new record:
        same words, refs = the replaced items (docs/contract.md 3-2)."""
        if outcome != "failure":
            raise ValueError("replaced_by goes with outcome='failure': the refs are what your record replaces")
        if not refs:
            raise ValueError("refs must name what your record replaces")
        found = self.runtime.events.get([record_id])
        if not found or found[0].kind != "record" or found[0].user != caller.user:
            raise ValueError(f"replaced_by must be the id of a record you wrote, got {record_id!r}")
        mine = found[0]
        id, _ = self._append(
            caller, kind="record", scope=mine.scope, text=mine.text, signal="self",
            conditions=dict(mine.conditions), refs=tuple(refs), repo=mine.repo, rev=mine.rev,
        )
        return id

    def recalled(self, since: datetime | None = None) -> list[str]:
        """Ids of what this user was handed in this repo since then and that is
        team knowledge now, oldest first. Read off the recall events, so it
        does not depend on the agent reporting anything. "In this repo" is the
        repo the handing session was configured with, whatever repo argument
        the agent passed; a session with no such note counts by its event's repo."""
        caller = self._caller()
        handed: dict[str, None] = {}
        for _, event in self.runtime.events.since(0):
            if event.kind != "recall" or event.user != self.user:
                continue
            if (self._sessions.get(f"session:{event.session}") or event.repo) != self.repo:
                continue
            if since is not None and event.at <= since:
                continue
            handed.update(dict.fromkeys(event.refs))
        known = {k.id for k in self.runtime.knowledge.get(list(handed)) if knowledge_visible(k, caller, "org")}
        return [id for id in handed if id in known]

    def _outcome_default(self, caller: Caller, refs: Sequence[str]) -> Scope:
        """An outcome is evidence about what it judges, so by default it is
        seen by the widest audience of the judged knowledge the caller can see.
        A team item can only be demoted by evidence the team sees."""
        scopes = [k.scope for k in self.runtime.knowledge.get(refs) if knowledge_visible(k, caller, "org")]
        # A record that is not knowledge yet is judged before its audience too.
        scopes += [e.scope for e in self.runtime.events.get(refs)
                   if e.kind == "record" and event_visible(e, caller, "org")]
        if not scopes:
            return self.policy.default_for(self.repo)
        return max(scopes, key=_WIDTH.__getitem__)


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


def service_from_env() -> Service:
    """A service for this process, configured by the environment variables above."""
    user, team = _identity()
    path = Path(os.environ.get("CORTEX_DB_PATH", "~/.cortex-mem/cortex.db")).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return Service(
        SqliteDatabase(path),
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
    )


def get_service() -> Service:
    global _service
    if _service is None:
        _service = service_from_env()
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
    again. commit_trailer: if a commit carries out what you recorded, put that
    line at the end of its message; the team's CI then reports the test result
    for your record, which is what turns it into team knowledge.
    When team knowledge on the same topic exists it comes back in
    related, with next saying what to do if your record replaces one of them."""
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
    scope: Scope | None = None,
    repo: str | None = None,
    replaced_by: str | None = None,
) -> dict:
    """Report whether what recalled memories say holds (refs = their ids, at
    least one). This judges the content, not how useful the recall was:
    "success": you did what the memory says and it worked as it says.
    "failure": what it says is wrong, or should no longer be done (e.g. it is a
    workaround and you fixed the cause).
    "unknown": you did not act on it. A memory that only pointed you to the
    right place is not a success; leave it unreported or use "unknown".
    What you found out belongs in record_episode, not in text here: text only
    explains the judgement and is not handed to anyone as a memory.
    Your report counts as your own word, also when you ran the tests yourself.
    Checks that count as outside evidence (CI, a review, a person) reach memory
    through the team's own pipeline, not through this tool.
    By default the judgement (outcome, refs) is visible to whoever can
    see the judged memories, e.g. the whole org for an org memory; scope can
    only narrow that. If that is wider than this repo's policy, text is not
    stored (text_redacted: true). Returns the id, the scope actually used,
    text_redacted, and delivered (false: stored, reaches memory later; do not
    report it again). replaced_by: with outcome "failure", the id of a record
    you wrote that replaces the memories in refs. They keep standing until your
    record passes an outside check."""
    try:
        # The signal is fixed here: an agent cannot vouch for itself as a test or a reviewer.
        return get_service().report_outcome(outcome, refs, text, "self", scope, repo, replaced_by)
    except ValueError as exc:  # e.g. empty refs: tell the agent why, not just that it failed
        raise ToolError(str(exc)) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    _identity()  # fail at start, not at the first tool call
    _max_failures()
    mcp.run()


if __name__ == "__main__":
    main()
