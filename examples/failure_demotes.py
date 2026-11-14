"""Live demo: a failing test demotes the memory the agent followed, so the
next recall no longer hands it off.

The agent never judges the memory itself (self reports are not independent
evidence). It recalls, answers, and records its answer with refs naming the
memories it followed. The demo's test step runs the answer and, like CI,
reports the result on the answer record and the memories it cites, so a
memory the agent was handed but did not follow is not blamed, and the wrong
answer does not live on as a memory of its own.

Three sessions share one SQLite file, each a separate cortex-mem-server process
over stdio (the same way an agent client starts it):
  A (alice) records an outdated command and an unrelated tip that still holds,
  B (bob's agent) recalls both, answers with the outdated command and cites it;
    the test step runs it, it fails, and the failure is reported on the answer and
    the memory it cites,
  C (carol) recalls again: the outdated command is gone, the tip is still there.

--mode scripted  B follows the memory about clearing the cache (no model).
--mode llm       an OpenAI-compatible model plays B with `recall` and
                 `record_episode`. Set
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
TIP = "dure 빌드가 느리면 빌드 캐시 크기부터 `dure cache stats`로 본다"
QUERY = "dure 빌드 캐시 비우기"
TASK = (
    "dure CLI로 빌드 캐시를 비우는 명령을 알려 줘. 먼저 팀 기억을 찾아봐. 답을 정했으면 record_episode로 "
    "답을 남기고, refs에는 네가 실제로 따른 기억의 id만 넣어. 마지막 줄에는 명령 한 줄만 백틱 없이 적어."
)

# The project's real CLI: `--purge` was replaced by `cache clear`.
CLI = """
import argparse, sys
p = argparse.ArgumentParser(prog="dure")
sub = p.add_subparsers(dest="cmd", required=True)
cache = sub.add_parser("cache").add_subparsers(dest="action", required=True)
cache.add_parser("clear")
cache.add_parser("stats")
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
    print("[A] alice records how she cleared the cache, and a tip")
    async with stdio_client(server("alice", db)) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        await call(s, "record_episode", text=MEMORY)
        await call(s, "record_episode", text=TIP)


async def agent_scripted(s: ClientSession, cited: list[str]) -> str:
    got = (await call(s, "recall", query=QUERY))["memories"]
    followed = next((m for m in got if "비우" in m["content"]), None)
    if followed is None:
        return ""
    command = re.search(r"`([^`]+)`", followed["content"]).group(1)
    out = await call(s, "record_episode", text=f"빌드 캐시는 {command}로 비웠다", refs=[followed["id"]])
    cited += [out["id"], followed["id"]]
    return command


def _refs(value: object) -> list[str]:
    # Some models send a list argument as a JSON string.
    if isinstance(value, str):
        value = json.loads(value) if value.startswith("[") else [value]
    return [str(v) for v in value] if isinstance(value, list) else []


async def agent_llm(s: ClientSession, cited: list[str], model: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL
    tools = [
        {"type": "function", "function": {"name": t.name, "description": t.description or "",
                                          "parameters": t.input_schema}}
        for t in (await s.list_tools()).tools
        if t.name in ("recall", "record_episode")
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
            if tc.function.name == "record_episode":
                args["refs"] = _refs(args.get("refs", []))
            out = await call(s, tc.function.name, **args)
            if tc.function.name == "record_episode":
                cited += [out["id"], *args["refs"]]
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})
    return ""


async def session_b(db: Path, cli: Path, model: str | None) -> None:
    print(f"[B] bob's agent ({model or 'scripted'}) answers; the test step runs the answer")
    async with stdio_client(server("bob", db)) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        cited: list[str] = []
        answer = await (agent_llm(s, cited, model) if model else agent_scripted(s, cited))
        passed = run_test(answer, cli)
        if not cited:
            print("  [test] the agent recorded no answer; nothing to judge")
            return
        # Reported through the same MCP server for the demo, so it is stored as
        # signal "self"; the engine demotes on a failure whoever judged it.
        print("  [test] reports the result on the agent's answer and the memories it cites")
        await call(s, "report_outcome", outcome="success" if passed else "failure", refs=sorted(set(cited)),
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
    gone = gone and not any("--purge" in m["content"] for m in left)
    kept = any(TIP in m["content"] for m in left)
    print("RESULT:", "PASS" if gone and kept else "FAIL",
          f"(outdated command gone: {gone}, uncited tip kept: {kept})")
    return 0 if gone and kept else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
