# Contributing

한국어: [CONTRIBUTING.ko.md](CONTRIBUTING.ko.md). This English file is the source of truth; a PR that changes one changes the other.

## Workflow

1. **Start from an issue.** Pick one or open one with the bug, feature or task form. A follow-up that comes out of a review is a task.
2. **Branch from `main`.** Name the branch `type/short-name`, for example `feat/evidence-engine`.
3. **Open a pull request** and fill in the template. Link the issue with `Fixes #123`, or `Part of #123` when the PR does not finish it.
4. **Merge by squash** once CI passes and a reviewer approves. The branch is deleted automatically.

## Pull requests

We squash-merge, so each PR becomes one commit on `main`: the PR title is the commit subject and the PR description is the commit body.

- **Title**: `type(scope): description`, for example `fix(server): reject an empty team`. The scope is optional. CI checks the format.
- **Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `build`, `chore`, `revert`.
- **Description**: write it for someone reading `git log` a year from now. Say what changes and why, and how you tested it. Put questions for the reviewer, screenshots and other discussion in PR comments, not in the description.
- **One PR, one change.** Leave unrelated cleanup for another PR.
- Write titles, descriptions and commit messages in English.

## Development

Python 3.12 or later and [uv](https://docs.astral.sh/uv/). The repository is one uv workspace with two packages: `cortex-mem` in `src/cortex_mem` and `cortex-mem-server` in `server/`. The [README](README.md) covers them and how to register the server.

```bash
uv sync             # install both packages
uv run pytest -q    # tests for both packages
```

CI runs the tests on Python 3.12, 3.13 and 3.14 on every PR. Run them before you push.

## The engine port contract

[docs/contract.md](docs/contract.md) is the contract between engines, the runtime and the harness. Its rules are part of the contract even where the types do not express them.

- A PR that changes the behavior of `contract.py`, `access.py` or `runtime.py` updates `docs/contract.md` in the same PR.
- [dure-ai/harness](https://github.com/dure-ai/harness) runs engines through the same runtime and pins a core commit. Say in the PR when a change needs the harness to follow.
- A new store implementation must pass the conformance tests in `cortex_mem.testing`.

## Use of AI

AI tools are welcome here, including agents that open issues and pull requests and write their descriptions.

- **You are responsible for what you submit.** Read the whole diff before you ask for review, be able to explain every change in it, and answer review comments yourself.
- **Disclose it in the PR.** Check "AI assistance was used to create this PR" in the template and name the tool. Because the PR description becomes the commit body, this is also the record in git history.
- **Do not credit a model in commits.** No `Co-Authored-By` or similar trailer for an AI tool. The repository's `.claude/settings.json` turns this off for Claude Code; if you use another tool, turn it off there.

Instructions for coding agents are in [AGENTS.md](AGENTS.md).
