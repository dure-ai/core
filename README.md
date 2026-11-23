# cortex-mem core

에이전트 기억을 Memory(소문)로 받고, 근거가 쌓이면 Knowledge(사실)로 승격하는 엔진의 포트 계약·런타임·저장소와 MCP 서버.

- 엔진 포트 계약: [docs/contract.md](docs/contract.md)
- 라이선스: Apache 2.0. 일부 코드는 같은 저자의 [mem-mcp](https://github.com/Jh-jaehyuk/mem-mcp)(MIT)에서 가져와 Apache 2.0으로 재라이선스했다. [NOTICE](NOTICE) 참고.

## 구성

uv workspace 하나에 패키지 두 개.

| 패키지 | 경로 | 내용 | 의존성 |
|---|---|---|---|
| `cortex-mem` | `src/cortex_mem` | 엔진 포트 계약, 런타임, 저장소(PostgreSQL, SQLite, 인메모리), 기준선 엔진, 저장소 적합성 테스트 | 없음 (`sqlite3`는 표준 라이브러리). PostgreSQL은 선택: `cortex-mem[postgres]` |
| `cortex-mem-server` | `server/` | MCP 서버: 설정을 읽어 저장소와 엔진을 조립하고, 도구 인자를 런타임 호출로 바꿈 | `cortex-mem[postgres]`, `mcp` |

하니스와 엔진은 `cortex-mem`에만 의존한다. 세 저장소는 같은 적합성 테스트(`cortex_mem.testing`)를 통과한다.

| 저장소 | 쓰임 | 비고 |
|---|---|---|
| PostgreSQL + pgvector | **기준 저장소**, 팀 서버 | 여러 세션·사람이 한 DB를 쓴다. 벡터 저장과 유사도 계산을 pgvector가 한다. 공식 측정과 운영이 같은 계산을 쓴다 |
| SQLite | 개인 로컬 | `CORTEX_DB_URL`이 없을 때의 기본. 파일 하나에 쓰기 잠금이 하나뿐이다 |
| 인메모리 | 시험·하니스 비교 | 프로세스가 끝나면 사라진다 |

PostgreSQL은 14 이상, pgvector는 정확 검색(기본)에 0.7 이상, `CORTEX_VECTOR_INDEX=hnsw`에 0.8 이상이 필요하다. 서버는 열 때 `vector` 타입이 있는지 확인하고(DB마다 `CREATE EXTENSION vector`, 없으면 시작할 때 멈춘다), hnsw를 켤 때는 pgvector가 0.8 이상인지도 검사한다(낮으면 시작할 때 멈춘다). PostgreSQL 버전과 정확 검색의 0.7 요구는 검사하지 않는다. 설치: `uv add "cortex-mem[postgres]"`. 서버는 이미 이것에 의존한다.

## 개발

```bash
uv sync
uv run python -m pytest -q
```

PostgreSQL 시험은 `CORTEX_TEST_PG_URL`이 있을 때만 돈다(없으면 건너뜀). 로컬 PG는 저장소 루트의 `compose.yaml`로 띄운다(pgvector 이미지, `127.0.0.1:54329`).

```bash
docker compose up -d pg
CORTEX_TEST_PG_URL=postgresql://postgres:cortex@localhost:54329/postgres uv run python -m pytest -q
```

이 비밀번호(`cortex`)는 로컬 시험 전용이다. 실제 DB에는 쓰지 않는다.

## Claude Code에 등록

```bash
claude mcp add cortex-mem -e CORTEX_TEAM=payments -- uv run --directory /path/to/core cortex-mem-server
```

`CORTEX_TEAM`은 필수다. 없으면 서버가 시작하지 않는다(잘못된 팀으로 쌓인 기록은 나중에 옮길 수 없다).

작업하는 저장소마다 `CORTEX_REPO`를 주려면, 그 저장소에서 local 범위로 등록한다(local 범위는 그 프로젝트에서만 쓰인다).

```bash
cd ~/work/api
claude mcp add cortex-mem -s local -e CORTEX_REPO=acme/api -e CORTEX_TEAM=payments \
  -- uv run --directory /path/to/core cortex-mem-server
```

서버는 `uv run --directory`로 core 폴더에서 돌기 때문에, 작업 디렉터리로는 에이전트가 어느 저장소에 있는지 알 수 없다. 그래서 저장소는 설정으로 준다.

도구: `record_episode`(기록), `recall`(불러오기), `report_outcome`(불러온 기억이 도움됐는지 보고). 서버 프로세스 하나가 세션 하나이고, 프로세스들은 SQLite 파일 하나 또는 PostgreSQL DB 하나를 함께 쓴다. 런타임이 이벤트를 엔진에 정확히 한 번 반영하므로(엔진 실행이 겹치면 한 결과만 남는다), 먼저 떠 있던 세션도 다른 세션의 기록을 본다.

| 환경변수 | 기본값 | |
|---|---|---|
| `CORTEX_DB_URL` | 없음 | `postgresql://...`. 있으면 PostgreSQL을 쓴다(비밀번호는 로그·오류에 나오지 않는다). 팀 서버에 권장 |
| `CORTEX_DB_PATH` | `~/.cortex-mem/cortex.db` | 이벤트 로그·지식 베이스·엔진 상태(SQLite). `CORTEX_DB_URL`이 없을 때만 쓰이고, 둘 다 있으면 경고를 남기고 PostgreSQL을 쓴다 |
| `CORTEX_DB_LOCK_TIMEOUT` | `30s` | PostgreSQL 쓰기 잠금 대기 한도(예: `30s`, `500ms`). 넘으면 분명한 오류 |
| `CORTEX_VECTOR_INDEX` | `none` | `none`(정확 검색) \| `hnsw`(근사 색인). 아래 참고 |
| `CORTEX_TEAM` | **필수** | 기록자의 팀. 없으면 서버가 시작하지 않는다 |
| `CORTEX_USER` | OS 사용자(경고를 남김) | 기록자 |
| `CORTEX_AGENT` | 없음 | 클라이언트 이름(예: `claude-code`) |
| `CORTEX_REPO` | 없음 | 이 프로젝트의 저장소(예: `acme/api`). 저장소별 공개 범위는 이 값으로만 정해진다. 기록·불러오기·결과 보고의 기본 `repo`(근거 위치, 질문한 곳)도 되며, 도구 인자 `repo`는 이 위치만 바꾸고 공개 범위는 바꾸지 못한다 |
| `CORTEX_ENGINE` | `promote_all` | `promote_all` \| `no_memory` \| `모듈:팩토리` |
| `CORTEX_RECORD_SCOPE` | `personal` | 기록의 기본 공개 범위 |
| `CORTEX_REPO_SCOPES` | 없음 | 저장소별 기본 범위. `acme/*=team,acme/handbook=org`처럼 쉼표로 구분, 먼저 맞는 것이 적용 |
| `CORTEX_MAX_FAILURES` | `3` | 엔진이 같은 이벤트에서 이만큼 실패하면 그 이벤트를 건너뛰고 로그를 남긴다 |

공개 범위는 에이전트가 아니라 서버 설정이 정한다. 에이전트는 `scope`로 기본값보다 좁히는 것만 할 수 있고(예: 팀 저장소에서 `personal` 메모), 넓히려 하면 기본값이 쓰인다. `recall`은 기본으로 볼 수 있는 범위 전체(본인·팀·조직)를 찾고, `repo`·`context`(예: `{"service": "api"}`)로 질문한 곳을 엔진에 알린다. `report_outcome`은 기본으로 판정 대상 지식을 볼 수 있는 범위에 남는다(팀 지식에 대한 실패 보고는 팀에 보여야 강등 근거가 된다). 대상이 지식 항목이 아니면 저장소 설정을 따른다. 그 범위가 저장소 설정보다 넓으면 판정(`outcome`·`signal`·`refs`)만 남고 에이전트가 쓴 `text`는 저장하지 않는다(응답과 이벤트의 `text_redacted`). 에이전트의 결과 보고는 `signal` 인자와 상관없이 자기 판정(`self`)으로 남는다(응답의 `signal`).

엔진이 실패해도 도구는 멈추지 않는다. 기록이 저장됐지만 엔진에 전달되지 못하면 `delivered: false`로 응답하고(다시 기록할 필요 없음, 나중에 전달됨), 같은 이벤트에서 `CORTEX_MAX_FAILURES`번 실패하면 그 이벤트를 건너뛰고 stderr에 남긴다.

`record_episode`와 `report_outcome`의 응답에는 `delivered`(엔진이 이 이벤트를 받았는지)와 `pending`(커서 뒤에 남은 이벤트 수, 다른 세션의 것 포함)이 있다. `pending`은 정수이고, 읽다가 실패하면 `null`이다(그 경우 `delivered`는 `false`이며, 기록은 이미 저장됐다).

## PostgreSQL

```bash
claude mcp add cortex-mem -e CORTEX_TEAM=payments \
  -e CORTEX_DB_URL=postgresql://user:password@host:5432/dbname \
  -- uv run --directory /path/to/core cortex-mem-server
```

- 쓰기 트랜잭션은 DB 전체 잠금 하나(advisory lock)로 직렬화하지만 짧다. 엔진 실행(`observe`·`tick`)과 임베딩은 트랜잭션 밖에서 돌고, 잠금 안에서는 읽은 것이 그대로인지 확인하고 쓰기만 한다. 위치는 커밋 순서대로 늘어난다. 잠금을 기다리는 세션은 `CORTEX_DB_LOCK_TIMEOUT`까지 기다린 뒤 분명한 오류를 낸다.
- 벡터는 pgvector `vector`(float32)로 저장하고 코사인 유사도는 DB 안에서 계산한다. 인메모리·SQLite와는 점수가 약 1e-6 이내에서 같다(차원이 커질수록 오차가 커진다. 4096차원에서 8e-7까지 측정).
- **HNSW는 기본 꺼져 있다.** 기본은 정확 검색이라 같은 데이터에서 실행마다 같은 결과가 나오고, 팀 규모(수천~수만 행)에서는 충분히 빠르다고 본다. `CORTEX_VECTOR_INDEX=hnsw`는 `search_events`·`search_knowledge`에만 쓰이고 묶기는 늘 정확 검색이다(묶음이 실행마다 달라지면 엔진 상태가 달라진다). 근사 색인은 진짜 최근접을 놓칠 수 있고 가시성 필터 때문에 k개보다 적게 돌려줄 수 있다. 차원이 4000을 넘으면 색인 없이 정확 검색만 한다. 색인은 halfvec(반정밀도)로 만들어지므로 결과가 근사다. HNSW 검색은 pgvector 0.8 이상이 필요하고, 열 때 검사한다. 색인 버전이 바뀌어 색인을 다시 만드는 동안(임베딩 모델이나 묶음 문턱값을 바꾼 뒤 첫 `sync`) HNSW 색인을 지우는 잠금이 재구축 트랜잭션 끝까지 남아, 다른 세션의 벡터 검색이 그동안 기다린다. 기본값은 지연을 잰 뒤 정한다.
- 연결이 끊기면(PostgreSQL 재시작, 유휴 연결 끊김 등) 다시 접속하지 않는다. 그 서버 프로세스는 세션을 다시 시작할 때까지 오류를 낸다.

기존 SQLite 파일을 빈 PostgreSQL DB로 옮기려면(원본은 바꾸지 않는다):

```bash
uv run python scripts/sqlite_to_pg.py ~/.cortex-mem/cortex.db postgresql://user:password@host:5432/dbname
```

이벤트(위치 그대로)·지식과 그 소유자·엔진 상태와 커서·임베딩 캐시를 옮기고, 벡터와 묶음은 옮기지 않는다. 임베딩을 켠 서버라면 옮긴 뒤 첫 `sync`가 색인을 다시 만들고(색인 버전에 저장소 종류가 들어 있다) 엔진을 처음부터 다시 전달한다. 임베딩은 캐시에서 오므로 API를 다시 부르지 않는다. 임베딩이 없으면 엔진은 옮긴 커서에서 이어 받는다. 대상은 새로 만든 DB 그대로여야 한다. 어느 표에든 행이 있거나 `cortex_meta`에 스키마 버전 말고 다른 값(예: 서버가 먼저 기록한 지식 베이스 소유자)이 있으면 종료 코드 2로 멈춘다. 원본이 float64 벡터를 담은 스키마 4(main의 스키마 4가 아닌, 이 기능의 예전 파일)이면 임베딩 캐시를 쓸 수 없어 다시 임베딩해야 한다는 경고를 stderr에 낸다. 옮기기가 끝나면 원본과 대상의 표별 행 수를 비교한 표를 출력한다.

## 임베딩

core는 `record` 이벤트와 지식 항목을 임베딩해 저장하고, 같은 (범위, 소유자) 안에서 비슷한 기록을 묶음(group)으로 묶는다. 엔진은 `ctx.similar`로 이 색인을 읽는다. 임베딩을 켜지 않으면 `ctx.similar`는 `None`이고 동작은 이전과 같다. 이벤트는 엔진에 전달되기 전에 색인된다(색인은 전달 앞의 별도 단계다). PostgreSQL이 기준 저장소이고, 인메모리·SQLite는 PostgreSQL과 같은 묶음을 만들며 점수는 약 1e-6 이내에서 같다(차원이 커질수록 오차가 커진다. 4096차원에서 8e-7까지 측정).

엔진 실행과 임베딩은 쓰기 트랜잭션 밖에서 돈다. 아직 색인되지 않은 기록의 본문은 최대 64개씩 묶어 임베딩해 캐시에 넣고, 승격된 지식 항목은 결정이 커밋된 뒤에 임베딩해 벡터를 따라가게 하며, 엔진이 `observe`나 `tick` 중에 부르는 `ctx.similar.search`·`ctx.similar.knowledge`의 질의도 쓰기 트랜잭션 밖에서 임베딩한다. 쓰기는 위치·세대를 확인하는 짧은 트랜잭션뿐이다. 그래서 느린 엔진이나 임베딩 API 장애가 다른 세션의 기록 저장을 막지 않는다. 장애 동안에는 전달만 멈추고 기록은 계속 저장되며, 장애가 풀리면 이어서 전달된다. 같은 이벤트에서 `observe`가 두 프로세스에서 겹쳐 실행될 수 있고 그중 한 결과만 반영되므로, 엔진은 `ctx.state` 말고는 바깥 상태를 바꾸지 않아야 한다(contract §3-1). 서버는 Upstage 요청 시간 제한을 4초로 둔다. 한 번의 요청이 멈춰 있을 수 있는 시간을 묶는 값이고(소켓 동작 하나마다의 제한이라, 응답이 아주 느리게 조금씩 오면 더 걸릴 수 있다), 잠금과는 관계가 없다.

| 환경변수 | 기본값 | |
|---|---|---|
| `CORTEX_EMBEDDER` | `none` | `none` \| `hash` \| `upstage` |
| `CORTEX_EMBED_PASSAGE_MODEL` | 없음 | Upstage 기록·지식용 모델. `upstage`일 때 필수 |
| `CORTEX_EMBED_QUERY_MODEL` | 없음 | Upstage 질의용 모델. `upstage`일 때 필수 |
| `UPSTAGE_API_KEY` | 없음 | `upstage`일 때 필수. 출력하지 않는다 |
| `CORTEX_GROUP_THRESHOLD` | `0.9` | 이 코사인 유사도 이상이면 같은 묶음 |

`HashEmbedder`(`hash`)는 단어를 해시로 세는 어휘 기반 특징 해싱이라 의미를 이해하지 못한다. 같은 단어를 쓰면 가깝고, 뜻이 같아도 단어가 다르면 멀다. 테스트와 오프라인 확인용이다. 모델이나 `CORTEX_GROUP_THRESHOLD`를 바꾸면 색인이 낡은 것으로 판정돼 다시 만들어지고(재구축) 엔진은 처음부터 다시 전달받는다(재생). 같은 DB(파일)를 쓰는 다른 엔진도 다음 호출 때 자기 상태가 옛 색인 위에서 만들어졌음을 알아채고 재생된다. 색인을 다시 만들면 세대가 오르므로 같은 설정으로 다시 만든 경우에도 모든 엔진이 다시 받는다. 한 저장소에는 색인 설정 하나와 엔진 버전(`state_version`) 하나만 쓴다: 설정이나 엔진 버전이 다른 서버 프로세스가 같은 DB(파일)를 쓰면 런타임이 이벤트마다 3번 다시 시도한 뒤 `IndexConfigChanged`나 `StateVersionChanged`로 크게 실패한다(지원하지 않는 구성). 설정이나 엔진을 바꿀 때는 그 DB(파일)를 쓰는 프로세스를 모두 내린 뒤 새 설정으로 띄운다. Claude Code 세션마다 서버 프로세스가 하나씩이므로, 그 저장소를 쓰는 모든 세션을 다시 시작한다는 뜻이다. 임베딩 변경이 처음 내보내는 스키마(`engine_cursor.epoch`, 지식 벡터의 `digest`)를 쓰지 않는 개발용 SQLite 파일과 PG 스키마는 지우고 시작한다.

엔진이 읽는 포트는 `SimilarityReader`다.

- `search(query, caller, scope, k, kinds)`: 질의와 가까운 이벤트
- `neighbors(event_id, caller, scope, k)`: 이벤트와 가까운 다른 이벤트
- `knowledge(query, caller, scope, k)`: 질의와 가까운 지식 항목
- `group(event_id)`: 이벤트가 속한 묶음(없으면 `None`)

점수는 코사인 유사도(-1~1)이고 텍스트 검색 점수와 섞어 비교하면 안 된다. 호출자가 볼 수 있는 범위만 결과에 나온다.

한계:

- 묶기는 single-link라서 A~B, B~C가 가까우면 A와 C가 멀어도 한 묶음이 될 수 있다.
- 기본 검색은 정확 검색이라 질의마다 볼 수 있는 범위 전체를 훑는다. 작은 팀에는 충분하다(PostgreSQL은 근사 색인을 켤 수 있다).
- 임베딩 서비스가 멈추면 전달이 멈춘다. `record_episode`·`report_outcome`은 기록을 저장하고 `delivered: false`로 답하며(건너뛰지 않고 장애가 풀리면 이어서 전달), `recall`은 지금까지 전달된 것으로 답한다(질문을 임베딩하지 못하면 텍스트 검색으로 답하고, 응답의 `pending`이 0보다 크다). 한 도구 호출 안에서는 임베딩을 한 번만 시도하고, Upstage 임베더는 장애 뒤 5초 동안(다시 실패할 때마다 두 배, 최대 60초) 요청하지 않고 바로 실패하므로, 장애 중 호출은 시간 제한을 매번 기다리지 않는다. 장애가 풀린 것은 그만큼 늦게 알아챌 수 있다. 장애는 시작과 끝에 한 줄씩만 로그에 남긴다. 서비스는 멀쩡한데 특정 텍스트만 거부되면(예: 모델 입력 한도를 넘는 긴 기록) 장애로 보지 않는다: 그 기록은 유사도 없이 전달되고 다른 기록을 막지 않으며, 거부된 질문은 유사도 결과 없이 답한다.
- 엔진의 상태 버전이 바뀌거나 색인을 다시 만들면(모델·`CORTEX_GROUP_THRESHOLD` 변경 등) 엔진은 처음부터 다시 전달받고, 그때 지식 베이스를 비운 뒤 다시 채운다. 더는 승격되지 않는 지식이 남지 않게 하기 위해서다. 다시 전달이 끝날 때까지는 지식이 일부만 보인다.
- 승격된 지식의 벡터는 결정이 커밋된 뒤에 따라온다. 프로세스가 죽거나 임베딩이 실패하면 잠깐 뒤처질 수 있고, 새 `Runtime`의 첫 `sync`나 색인 재구축 뒤에 전체 비교로 따라잡는다(이미 돌고 있는 프로세스는 다시 훑지 않는다). 대기 중인 지식을 임베딩하지 못하는 동안 `sync`는 이벤트를 전달하기 전에 오류를 낸다.
- 이벤트 하나를 전달하는 데 쓰기 트랜잭션이 6개 든다(기록 추가, 이벤트 본문 캐시, 색인 위치·벡터·묶음, 전달 커밋, 지식 본문 캐시, 지식 벡터). 엔진 실행과 임베딩을 한 트랜잭션에 넣던 이전 구조(4개)보다 많지만 각각 짧고, 느린 엔진이 있어도 다른 세션의 기록이 몇 밀리초 안에 저장된다(엔진이 3초 멈춘 동안 다른 연결의 기록 저장 대기: 2.7초 → SQLite 2ms, PG 12ms. `scripts/bench_delivery.py`로 잰다).
- 옛 저장소: 이 변경 이전의 SQLite 파일은 열 때 `engine_cursor.epoch`와 `vectors.digest` 열이 자동으로 더해진다(main의 스키마 3·4 파일도 그대로 열리고, 검색 색인은 지금의 토큰으로 다시 만든다). 이 변경 이전의 PG 개발 스키마는 두 열이 없어서 열 때 분명한 오류로 멈추니 스키마를 지우고 다시 만든다(PG 저장소는 이 변경과 함께 처음 나간다).
- 삭제는 아직 지원하지 않는다.

## 예시: 테스트 실패가 기억을 강등한다

[examples/failure_demotes.py](examples/failure_demotes.py)는 서버 프로세스 셋(세션 셋)이 SQLite 파일 하나를 함께 쓰며, 낡은 명령을 담은 기억이 테스트 실패 뒤 다음 `recall`에서 빠지는 흐름을 보인다.

1. alice가 낡은 명령(`dure cache --purge`)과 아직 맞는 팁(`dure cache stats`)을 기록한다.
2. bob의 에이전트가 `recall`로 둘 다 받고, 낡은 명령으로 답한 뒤 `record_episode`로 답을 기록한다. 이때 `refs`에 따른 기억의 id를 적는다.
3. 테스트 단계가 그 답을 실행하고, 실패를 답 기록과 `refs`의 기억에만 `report_outcome`으로 보고한다. 에이전트는 판정하지 않는다.
4. carol이 다시 `recall`하면 낡은 명령은 빠지고, 인용되지 않은 팁은 남는다.

데모의 테스트 단계는 MCP 서버를 거쳐 보고하므로 판정은 `signal="self"`로 저장된다. 엔진은 실패 보고면 신호와 상관없이 강등한다. 진짜 `test` 신호는 MCP 밖의 입력 경로(CI 명령 등)가 생기면 쓴다.

core 폴더에서 실행한다. 마지막 줄이 `RESULT: PASS ...`이면 성공이다.

```bash
uv run python examples/failure_demotes.py                       # 모델 없이
OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.upstage.ai/v1 \
  uv run --with openai python examples/failure_demotes.py --mode llm --model solar-pro4
```

모델 모드(`--mode llm`)만 API 키와 `--with openai`가 필요하고, 호출 비용이 든다. solar-pro4로 한 번 돌리면 3턴, 입력 약 3,300 / 출력 약 100 토큰이다.
