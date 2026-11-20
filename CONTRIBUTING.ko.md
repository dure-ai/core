# 기여 안내

English: [CONTRIBUTING.md](CONTRIBUTING.md). 영어 문서가 기준입니다. 한쪽을 고치는 PR은 다른 쪽도 같이 고칩니다.

## 작업 흐름

1. **이슈에서 시작합니다.** 기존 이슈를 고르거나 bug, feature, task 양식으로 새로 엽니다. 리뷰에서 나온 후속 작업은 task입니다.
2. **`main`에서 브랜치를 만듭니다.** 이름은 `type/short-name` 형식입니다. 예: `feat/evidence-engine`
   아직 머지되지 않은 다른 브랜치 위에 쌓은 브랜치는, 그 브랜치가 스쿼시 머지된 뒤 옮깁니다: `git rebase --onto origin/main <base-branch> <your-branch>`. `main`에는 base 브랜치의 커밋들 대신 스쿼시 커밋 하나가 들어가므로, PR의 base만 바꾸면 그 커밋들이 내 PR에 남습니다.
3. **PR을 열고** 템플릿을 채웁니다. 이슈는 `Fixes #123`으로 연결하고, PR 하나로 이슈가 끝나지 않으면 `Part of #123`을 씁니다.
4. CI가 통과하고 리뷰어가 승인하면 **스쿼시로 머지합니다.** 브랜치는 자동으로 삭제됩니다.

## Pull request

스쿼시 머지를 쓰므로 PR 하나가 `main`의 커밋 하나가 됩니다. PR 제목이 커밋 제목이 되고, PR 설명이 커밋 본문이 됩니다.

- **제목**: `type(scope): description` 형식입니다. 예: `fix(server): reject an empty team`. scope는 생략할 수 있습니다. CI가 형식을 검사합니다.
- **type**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `build`, `chore`, `revert`.
- **설명**: 1년 뒤 `git log`를 읽는 사람을 위해 씁니다. 무엇을 왜 바꿨는지, 어떻게 확인했는지 적습니다. 리뷰어에게 묻는 질문, 스크린샷, 그 밖의 논의는 설명이 아니라 PR 코멘트에 씁니다.
- **PR 하나에 변경 하나.** 관련 없는 정리는 다른 PR로 보냅니다.
- 제목, 설명, 커밋 메시지는 영어로 씁니다.

## 개발

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다. 저장소는 uv workspace 하나에 패키지 두 개입니다: `src/cortex_mem`의 `cortex-mem`과 `server/`의 `cortex-mem-server`. 패키지 설명과 서버 등록 방법은 [README](README.md)에 있습니다.

```bash
uv sync             # 두 패키지 설치
uv run pytest -q    # 두 패키지의 테스트
```

CI가 모든 PR에서 Python 3.12, 3.13, 3.14로 테스트를 돌립니다. 푸시하기 전에 직접 돌려 보세요.

## 엔진 포트 계약

[docs/contract.md](docs/contract.md)는 엔진, 런타임, 하니스 사이의 계약입니다. 타입으로 표현되지 않는 규칙도 계약의 일부입니다.

- `contract.py`, `access.py`, `runtime.py`의 동작을 바꾸는 PR은 같은 PR에서 `docs/contract.md`도 고칩니다.
- [dure-ai/harness](https://github.com/dure-ai/harness)는 같은 런타임으로 엔진을 돌리고 core 커밋을 고정해서 씁니다. 하니스가 따라와야 하는 변경이면 PR에 적습니다.
- 새 저장소 구현은 `cortex_mem.testing`의 적합성 테스트를 통과해야 합니다.

## AI 사용

AI 도구를 써도 됩니다. 에이전트가 이슈와 PR을 열고 설명을 쓰는 것도 괜찮습니다.

- **제출한 사람이 책임집니다.** 리뷰를 요청하기 전에 diff 전체를 읽고, 그 안의 모든 변경을 설명할 수 있어야 하며, 리뷰 코멘트에는 직접 답합니다.
- **PR에 표기합니다.** 템플릿의 "AI assistance was used to create this PR"에 체크하고 도구 이름을 적습니다. PR 설명이 커밋 본문이 되므로 이 표기가 git 기록에도 남습니다.
- **커밋에 모델을 올리지 않습니다.** AI 도구에 대한 `Co-Authored-By` 같은 트레일러를 넣지 않습니다. Claude Code는 저장소의 `.claude/settings.json`이 꺼 줍니다. 다른 도구를 쓴다면 그 도구에서 끄세요.

코딩 에이전트용 지시는 [AGENTS.md](AGENTS.md)에 있습니다.
