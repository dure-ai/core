"""cortex-mem: the entry for evidence that is not an agent's own word.

The MCP tools record what agents say (signal "self"). A test result or a
person's judgement comes in here instead, run by CI, by a git hook or by a
person. None of it depends on what an agent chose to report.

    cortex-mem confirm-commits <rev-range> --outcome success|failure
        For each commit in the range, read the trailers
        "Cortex-Record: <record id>" and report the outcome of the test run
        for those records, with the commit as its rev. Run it in the
        repository CI just tested.

    cortex-mem install-hook
        Install a commit-msg hook in this repository. On every commit it adds
        "Cortex-Recalled: <id>" for each piece of team knowledge the
        committer was handed in this repo since the previous commit. The
        hook keeps the configuration of the shell that installs it.

    cortex-mem to-review <rev-range>
        The knowledge named in Cortex-Recalled trailers of the range, with
        its text and the commits made after it was handed over: what a
        reviewer of those commits should look at.

    cortex-mem review <id> keep|reject [--text WHY]
    cortex-mem review <id> replace --commit <sha>
        A person's judgement. keep: it still holds. reject: it does not.
        replace: the commit makes it obsolete; the commit message becomes a
        record the reviewer vouches for, and the old item is rejected.
        CORTEX_USER is the reviewer. (approve is accepted for keep.)

Configuration is the server's (CORTEX_TEAM, CORTEX_USER, CORTEX_DB_PATH,
CORTEX_ENGINE, CORTEX_REPO, ...). Prints one JSON object.

Nothing here checks who runs it: whoever can write the database can. Keep
this command out of what agents are allowed to run.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .server import RECALLED_TRAILER, TRAILER, Service, service_from_env

_HOOK_MARK = "# installed by cortex-mem install-hook"
_ENV = ("CORTEX_TEAM", "CORTEX_USER", "CORTEX_DB_PATH", "CORTEX_ENGINE", "CORTEX_REPO", "CORTEX_RECORD_SCOPE",
        "CORTEX_REPO_SCOPES")


def _git(*args: str, cwd: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _trailers(name: str, rev_range: str, cwd: str | None = None) -> list[tuple[str, str, list[str]]]:
    """(sha, subject, values of trailer `name`), oldest first, for commits that carry it."""
    pattern = re.compile(rf"^{name}:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE)
    out = _git("log", "--reverse", "--format=%H%x00%B%x01", rev_range, "--", cwd=cwd)
    found = []
    for entry in filter(None, (e.strip("\n") for e in out.split("\x01"))):
        sha, _, message = entry.partition("\x00")
        values = list(dict.fromkeys(pattern.findall(message)))
        if values:
            found.append((sha, message.strip().splitlines()[0], values))
    return found


def commit_records(rev_range: str, cwd: str | None = None) -> list[tuple[str, list[str]]]:
    """(commit sha, record ids named in its trailers), oldest first, for commits that name any."""
    return [(sha, ids) for sha, _, ids in _trailers(TRAILER, rev_range, cwd)]


def confirm_commits(service: Service, rev_range: str, outcome: str, cwd: str | None = None, text: str = "") -> dict:
    reported, unknown = [], []
    for sha, ids in commit_records(rev_range, cwd):
        known = {e.id for e in service.runtime.events.get(ids) if e.kind == "record"}
        unknown += [i for i in ids if i not in known]
        refs = [i for i in ids if i in known]
        if refs:
            result = service.report_outcome(outcome, refs, text, signal="test", rev=sha)  # type: ignore[arg-type]
            reported.append({"commit": sha, "records": refs, "outcome": outcome, "id": result["id"]})
    return {"reported": reported, "unknown_records": unknown}


def _amending(cwd: str | None = None) -> bool:
    """Whether the commit being made rewrites HEAD. Git does not tell a
    commit-msg hook, but it hands it the author date, and an amended commit
    keeps the one it had. (`--amend --reset-author` therefore looks like a
    new commit, and so does a commit made in the same second as HEAD.)"""
    stamp = re.match(r"@(\d+)", os.environ.get("GIT_AUTHOR_DATE", ""))
    try:
        return bool(stamp) and stamp.group(1) == _git("log", "-1", "--format=%at", cwd=cwd).strip()
    except subprocess.CalledProcessError:
        return False


def add_recalled(service: Service, message_file: str, cwd: str | None = None) -> list[str]:
    """Add a Cortex-Recalled trailer for what the committer was handed since
    the previous commit: HEAD, or HEAD's parent when HEAD is being amended,
    so that rewriting a commit's message does not drop what it named."""
    previous = "HEAD~1" if _amending(cwd) else "HEAD"
    try:
        since: datetime | None = datetime.fromisoformat(_git("log", "-1", "--format=%cI", previous, cwd=cwd).strip())
    except (subprocess.CalledProcessError, ValueError):
        since = None  # first commit of the repository
    ids = service.recalled(since)
    if ids:
        args = [a for id in ids for a in ("--trailer", f"{RECALLED_TRAILER}: {id}")]
        _git("interpret-trailers", "--in-place", "--if-exists", "addIfDifferent", *args, message_file, cwd=cwd)
    return ids


def install_hook(cwd: str | None = None) -> dict:
    hooks = Path(_git("rev-parse", "--git-path", "hooks", cwd=cwd).strip())
    if not hooks.is_absolute():
        hooks = Path(cwd or ".") / hooks
    hook = hooks / "commit-msg"
    if hook.exists() and _HOOK_MARK not in hook.read_text():
        raise SystemExit(f"cortex-mem: {hook} exists and is not ours; add `cortex-mem commit-msg \"$1\"` to it yourself")
    exe = shutil.which("cortex-mem") or sys.argv[0]
    env = "".join(f"export {k}={shlex.quote(os.environ[k])}\n" for k in _ENV if os.environ.get(k))
    hooks.mkdir(parents=True, exist_ok=True)
    # A failure here must never stop a commit.
    hook.write_text(f"#!/bin/sh\n{_HOOK_MARK}\n{env}{shlex.quote(str(Path(exe).resolve()))} commit-msg \"$1\" "
                    ">/dev/null 2>&1 || true\n")
    hook.chmod(0o755)
    return {"installed": str(hook)}


def to_review(service: Service, rev_range: str, cwd: str | None = None) -> dict:
    commits: dict[str, list[dict]] = {}
    for sha, subject, ids in _trailers(RECALLED_TRAILER, rev_range, cwd):
        for id in ids:
            commits.setdefault(id, []).append({"commit": sha, "subject": subject})
    items = {k.id: k for k in service.runtime.knowledge.get(list(commits))}
    reports: dict[str, list[dict]] = {}
    for _, e in service.runtime.events.since(0):
        if e.kind == "outcome" and (e.signal or "self") == "self":
            for ref in e.refs:
                if ref in commits:
                    reports.setdefault(ref, []).append({"by": e.user, "outcome": e.outcome, "text": e.text})
    return {
        "to_review": [{"id": id, "content": items[id].body, "commits": commits[id],
                       "agent_reports": reports.get(id, [])[-3:]} for id in commits if id in items],
        "no_longer_knowledge": [id for id in commits if id not in items],
    }


def review(service: Service, ref: str, verdict: str, text: str = "", commit: str | None = None,
           cwd: str | None = None) -> dict:
    if verdict == "replace":
        if not commit:
            raise SystemExit("cortex-mem: review ... replace needs --commit <sha>")
        sha = _git("rev-parse", commit, cwd=cwd).strip()
        message = _git("log", "-1", "--format=%B", sha, cwd=cwd)
        body = re.sub(rf"^({TRAILER}|{RECALLED_TRAILER}|Co-Authored-By):.*$\n?", "", message,
                      flags=re.MULTILINE | re.IGNORECASE).strip()
        old = service.runtime.knowledge.get([ref])
        new = service.record(body, scope=old[0].scope if old else None, refs=[ref], rev=sha, signal="human")
        out = service.report_outcome("failure", [ref], text or f"replaced by commit {sha[:12]}", signal="human", rev=sha)
        return {"rejected": ref, "outcome": out["id"], "record": new["id"], "commit": sha}
    outcome = "success" if verdict in ("keep", "approve") else "failure"
    return service.report_outcome(outcome, [ref], text, signal="human")  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cortex-mem", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cc = sub.add_parser("confirm-commits", help="report a test run for the records named in commit trailers")
    cc.add_argument("rev_range")
    cc.add_argument("--outcome", required=True, choices=["success", "failure"])
    cc.add_argument("--text", default="")
    sub.add_parser("install-hook", help="install the commit-msg hook in this repository")
    cm = sub.add_parser("commit-msg", help="(hook) add Cortex-Recalled trailers to a commit message file")
    cm.add_argument("message_file")
    tr = sub.add_parser("to-review", help="knowledge handed over before the commits of a range")
    tr.add_argument("rev_range")
    rv = sub.add_parser("review", help="a person's judgement of one record or memory")
    rv.add_argument("ref")
    rv.add_argument("verdict", choices=["keep", "approve", "reject", "replace"])
    rv.add_argument("--text", default="")
    rv.add_argument("--commit")
    args = parser.parse_args(argv)
    if args.command == "install-hook":
        result: dict = install_hook()
    else:
        service = service_from_env()
        if args.command == "confirm-commits":
            result = confirm_commits(service, args.rev_range, args.outcome, text=args.text)
        elif args.command == "commit-msg":
            result = {"recalled": add_recalled(service, args.message_file)}
        elif args.command == "to-review":
            result = to_review(service, args.rev_range)
        else:
            result = review(service, args.ref, args.verdict, args.text, args.commit)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
