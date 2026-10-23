# cortex-mem MCP server

에이전트 기억을 Memory(소문)로 받고, 근거가 쌓이면 Knowledge(사실)로 승격하는 엔진의 MCP 서버와 엔진 포트 계약.

- 엔진 포트 계약: [docs/contract.md](docs/contract.md)
- 라이선스: Apache 2.0. 일부 코드는 같은 저자의 [mem-mcp](https://github.com/Jh-jaehyuk/mem-mcp)(MIT)에서 가져와 Apache 2.0으로 재라이선스했다. [NOTICE](NOTICE) 참고.

## 구성

uv workspace 하나에 패키지 두 개.

| 패키지 | 경로 | 내용 | 의존성 |
|---|---|---|---|
| `cortex-mem` | `src/cortex_mem` | 엔진 포트 계약, 런타임, 저장소(인메모리, SQLite), 기준선 엔진, 저장소 적합성 테스트 | 없음 (`sqlite3`는 표준 라이브러리) |
| `cortex-mem-server` | `server/` | MCP 서버: 설정을 읽어 저장소와 엔진을 조립하고, 도구 인자를 런타임 호출로 바꿈 | `cortex-mem`, `mcp` |

하니스와 엔진은 `cortex-mem`에만 의존한다. SQLite 저장소는 PostgreSQL 구현이 나오기 전까지의 임시 구현이고, PG 구현도 같은 적합성 테스트(`cortex_mem.testing`)를 통과해야 한다.

## 개발

```bash
uv sync
uv run pytest
```

## Claude Code에 등록

```bash
claude mcp add cortex-mem -e CORTEX_TEAM=payments -- uv run --directory /path/to/mcp-server cortex-mem-server
```

`CORTEX_TEAM`은 필수다. 없으면 서버가 시작하지 않는다(잘못된 팀으로 쌓인 기록은 나중에 옮길 수 없다).

작업하는 저장소마다 `CORTEX_REPO`를 주려면, 그 저장소에서 local 범위로 등록한다(local 범위는 그 프로젝트에서만 쓰인다).

```bash
cd ~/work/api
claude mcp add cortex-mem -s local -e CORTEX_REPO=acme/api -e CORTEX_TEAM=payments \
  -- uv run --directory /path/to/mcp-server cortex-mem-server
```

서버는 `uv run --directory`로 mcp-server 폴더에서 돌기 때문에, 작업 디렉터리로는 에이전트가 어느 저장소에 있는지 알 수 없다. 그래서 저장소는 설정으로 준다.

도구: `record_episode`(기록), `recall`(불러오기), `report_outcome`(불러온 기억이 도움됐는지 보고). 서버 프로세스 하나가 세션 하나이고, 프로세스들은 SQLite 파일 하나를 함께 쓴다. 런타임이 이벤트를 엔진에 정확히 한 번 전달하므로, 먼저 떠 있던 세션도 다른 세션의 기록을 본다.

| 환경변수 | 기본값 | |
|---|---|---|
| `CORTEX_DB_PATH` | `~/.cortex-mem/cortex.db` | 이벤트 로그·지식 베이스·엔진 상태(SQLite) |
| `CORTEX_TEAM` | **필수** | 기록자의 팀. 없으면 서버가 시작하지 않는다 |
| `CORTEX_USER` | OS 사용자(경고를 남김) | 기록자 |
| `CORTEX_AGENT` | 없음 | 클라이언트 이름(예: `claude-code`) |
| `CORTEX_REPO` | 없음 | 이 프로젝트의 저장소(예: `acme/api`). 저장소별 공개 범위는 이 값으로만 정해진다. 기록·불러오기·결과 보고의 기본 `repo`(근거 위치, 질문한 곳)도 되며, 도구 인자 `repo`는 이 위치만 바꾸고 공개 범위는 바꾸지 못한다 |
| `CORTEX_ENGINE` | `promote_all` | `promote_all` \| `no_memory` \| `모듈:팩토리` |
| `CORTEX_RECORD_SCOPE` | `personal` | 기록의 기본 공개 범위 |
| `CORTEX_REPO_SCOPES` | 없음 | 저장소별 기본 범위. `acme/*=team,acme/handbook=org`처럼 쉼표로 구분, 먼저 맞는 것이 적용 |
| `CORTEX_MAX_FAILURES` | `3` | 엔진이 같은 이벤트에서 이만큼 실패하면 그 이벤트를 건너뛰고 로그를 남긴다 |

공개 범위는 에이전트가 아니라 서버 설정이 정한다. 에이전트는 `scope`로 기본값보다 좁히는 것만 할 수 있고(예: 팀 저장소에서 `personal` 메모), 넓히려 하면 기본값이 쓰인다. `recall`은 기본으로 볼 수 있는 범위 전체(본인·팀·조직)를 찾고, `repo`·`context`(예: `{"service": "api"}`)로 질문한 곳을 엔진에 알린다. `report_outcome`은 기본으로 판정 대상 지식을 볼 수 있는 범위에 남는다(팀 지식에 대한 실패 보고는 팀에 보여야 강등 근거가 된다). 대상이 지식 항목이 아니면 저장소 설정을 따른다. 그 범위가 저장소 설정보다 넓으면 판정(`outcome`·`signal`·`refs`)만 남고 에이전트가 쓴 `text`는 저장하지 않는다(응답과 이벤트의 `text_redacted`).

엔진이 실패해도 도구는 멈추지 않는다. 기록이 저장됐지만 엔진에 전달되지 못하면 `delivered: false`로 응답하고(다시 기록할 필요 없음, 나중에 전달됨), 같은 이벤트에서 `CORTEX_MAX_FAILURES`번 실패하면 그 이벤트를 건너뛰고 stderr에 남긴다.
