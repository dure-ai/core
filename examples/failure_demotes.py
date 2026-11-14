"""Live demo: a failing test demotes the memory the agent followed, so the
next recall no longer hands it off.

The agent never judges the memory itself (self reports are not independent
evidence). It recalls and answers; the demo's test step runs the answer and,
like CI, reports the result with signal="test" on the memories it was handed.

Three sessions share one SQLite file, each a separate cortex-mem-server process
over stdio (the same way an agent client starts it):
  A (alice) records an outdated command,
  B (bob's agent) recalls it and answers with it; the test step runs it, it fails,
    the test step reports failure on the recalled memories,
  C (carol) recalls again and the outdated command is gone.

--mode scripted  B answers with the command in the first recalled memory (no model).
--mode llm       an OpenAI-compatible model plays B with only `recall`. Set
                 OPENAI_API_KEY, and OPENAI_BASE_URL for a non-OpenAI server
                 (e.g. https://api.upstage.ai/v1 with --model solar-pro4).

Run from the core folder:
  uv run python examples/failure_demotes.py
  uv run --with openai python examples/failure_demotes.py --mode llm --model solar-pro4
"""

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ENGINE = "cortex_mem.engines.evidence:EvidenceEngine"

MEMORY = "dure CLI에서 빌드 캐시를 비우려면 `dure cache --purge`를 실행한다"
QUERY = "dure 빌드 캐시 비우기"
TASK = "dure CLI로 빌드 캐시를 비우는 명령을 알려 줘. 먼저 팀 기억을 찾아보고, 마지막 줄에 명령 한 줄만 백틱 없이 적어."

# The project's real CLI: `--purge` was replaced by `cache clear`.
CLI = """
import argparse, sys
p = argparse.ArgumentParser(prog="dure")
sub = p.add_subparsers(dest="cmd", required=True)
cache = sub.add_parser("cache").add_subparsers(dest="action", required=True)
cache.add_parser("clear")
p.parse_args()
print("cache cleared")
"""


def server(user: str, db: Path) -> StdioServerParameters:
    env = {
        "PATH": os.environ["PATH"],
        "CORTEX_TEAM": "dure",
        "CORTEX_USER": user,
        "CORTEX_DB_PATH": str(db),
        "CORTEX_ENGINE": ENGINE,
        "CORTEX_RECORD_SCOPE": "team",
    }
    # The same interpreter, so the server runs from this checkout's environment.
    return StdioServerParameters(command=sys.executable, args=["-m", "cortex_mem_server.server"], env=env)


def payload(result) -> dict:
    if result.is_error:
        raise RuntimeError(result.content)
    return result.structured_content or json.loads(result.content[0].text)


async def call(session: ClientSession, tool: str, **args) -> dict:
    out = payload(await session.call_tool(tool, args))
    print(f"  {tool}({json.dumps(args, ensure_ascii=False)})\n    -> {json.dumps(out, ensure_ascii=False)}")
    return out


def run_test(answer: str, cli: Path) -> bool:
    """The test step: run the agent's command against the real CLI. Only a
    `dure ...` command is run, as an argument list, never through a shell."""
    line = answer.strip().splitlines()[-1].strip().strip("`") if answer.strip() else ""
    argv = shlex.split(line)
    if not argv or argv[0] != "dure":
        print(f"  [test] not a dure command: {line!r}")
        return False
    done = subprocess.run([sys.executable, str(cli), *argv[1:]], capture_output=True, text=True, check=False)
    print(f"  [test] $ {line}\n    exit {done.returncode}: {(done.stdout or done.stderr).strip().splitlines()[-1]}")
    return done.returncode == 0


async def session_a(db: Path) -> None:
    print("[A] alice records how she cleared the cache")
    async with stdio_client(server("alice", db)) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        await call(s, "record_episode", text=MEMORY)


async def agent_scripted(s: ClientSession, recalled: list[str]) -> str:
    got = (await call(s, "recall", query=QUERY))["memories"]
    recalled += [m["id"] for m in got]
    found = re.search(r"`([^`]+)`", got[0]["content"]) if got else None
    return found.group(1) if found else ""


async def agent_llm(s: ClientSession, recalled: list[str], model: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL
    tools = [
        {"type": "function", "function": {"name": t.name, "description": t.description or "",
                                          "parameters": t.input_schema}}
        for t in (await s.list_tools()).tools
        if t.name == "recall"
    ]
    messages = [
        {"role": "system", "content": "너는 코딩 에이전트다. 작업 전에 recall로 팀 기억을 찾아라."},
        {"role": "user", "content": TASK},
    ]
    for _ in range(4):
        resp = await client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=400)
        print(f"  [usage] {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out")
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            print(f"  agent: {msg.content}")
            return msg.content or ""
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = await call(s, tc.function.name, **args)
            recalled += [m["id"] for m in out["memories"]]
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})
    return ""


async def session_b(db: Path, cli: Path, model: str | None) -> None:
    print(f"[B] bob's agent ({model or 'scripted'}) answers; the test step runs the answer")
    async with stdio_client(server("bob", db)) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        recalled: list[str] = []
        answer = await (agent_llm(s, recalled, model) if model else agent_scripted(s, recalled))
        passed = run_test(answer, cli)
        if recalled:
            print("  [test] reports the result on the memories the agent was handed")
            await call(s, "report_outcome", outcome="success" if passed else "failure",
                       refs=sorted(set(recalled)), signal="test",
                       text="" if passed else "dure cache --purge: 없는 옵션이라 실패. 지금은 dure cache clear")


async def session_c(db: Path) -> list:
    print("[C] carol recalls the same thing")
    async with stdio_client(server("carol", db)) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        return (await call(s, "recall", query=QUERY))["memories"]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--model", default="solar-pro4")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        db, cli = Path(tmp) / "demo.db", Path(tmp) / "dure_cli.py"
        cli.write_text(CLI)
        await session_a(db)
        await session_b(db, cli, args.model if args.mode == "llm" else None)
        left = await session_c(db)
    gone = not any(MEMORY in m["content"] for m in left)
    print("RESULT:", "PASS, the outdated command is no longer handed off" if gone else "FAIL, carol still gets it")
    return 0 if gone else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
