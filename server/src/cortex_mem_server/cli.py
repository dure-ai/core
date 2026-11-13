"""cortex-mem: the entry for evidence that is not an agent's own word.

The MCP tools record what agents say (signal "self"). A test result or a
person's judgement comes in here instead, run by CI or by a person:

    cortex-mem confirm-commits <rev-range> --outcome success|failure
        For each commit in the range (git rev-list), read the trailers
        "Cortex-Record: <record id>" in its message and report the outcome of
        the test run for those records, with the commit as its rev.
        Run it in the repository CI just tested, e.g.
            cortex-mem confirm-commits "$BASE..$HEAD" --outcome success

    cortex-mem review <record or memory id> approve|reject [--text WHY]
        A person's judgement of one record. CORTEX_USER is the reviewer.

Configuration is the server's (CORTEX_TEAM, CORTEX_USER, CORTEX_DB_PATH,
CORTEX_ENGINE, CORTEX_REPO, ...). Prints one JSON object.

Nothing here checks who runs it: whoever can write the database can. Keep
this command out of what agents are allowed to run.
"""

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence

from .server import TRAILER, Service, service_from_env

_TRAILER = re.compile(rf"^{TRAILER}:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


def commit_records(rev_range: str, cwd: str | None = None) -> list[tuple[str, list[str]]]:
    """(commit sha, record ids named in its trailers), oldest first, for commits that name any."""
    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%x00%B%x01", rev_range, "--"],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout
    found = []
    for entry in filter(None, (e.strip("\n") for e in out.split("\x01"))):
        sha, _, message = entry.partition("\x00")
        ids = list(dict.fromkeys(_TRAILER.findall(message)))
        if ids:
            found.append((sha, ids))
    return found


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


def review(service: Service, ref: str, verdict: str, text: str = "") -> dict:
    outcome = "success" if verdict == "approve" else "failure"
    return service.report_outcome(outcome, [ref], text, signal="human")  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cortex-mem", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cc = sub.add_parser("confirm-commits", help="report a test run for the records named in commit trailers")
    cc.add_argument("rev_range")
    cc.add_argument("--outcome", required=True, choices=["success", "failure"])
    cc.add_argument("--text", default="")
    rv = sub.add_parser("review", help="a person's judgement of one record")
    rv.add_argument("ref")
    rv.add_argument("verdict", choices=["approve", "reject"])
    rv.add_argument("--text", default="")
    args = parser.parse_args(argv)
    service = service_from_env()
    if args.command == "confirm-commits":
        result = confirm_commits(service, args.rev_range, args.outcome, text=args.text)
    else:
        result = review(service, args.ref, args.verdict, args.text)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
