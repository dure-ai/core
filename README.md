# cortex-mem core

에이전트 기억을 Memory(소문)로 받고, 근거가 쌓이면 Knowledge(사실)로 승격하는 엔진의 포트 계약·런타임·저장소와 MCP 서버.

- 엔진 포트 계약: [docs/contract.md](docs/contract.md)
- 라이선스: Apache 2.0. 일부 코드는 같은 저자의 [mem-mcp](https://github.com/Jh-jaehyuk/mem-mcp)(MIT)에서 가져와 Apache 2.0으로 재라이선스했다. [NOTICE](NOTICE) 참고.

## 구성

uv workspace 하나에 패키지 두 개.

| 패키지 | 경로 | 내용 | 의존성 |
|---|---|---|---|
| `cortex-mem` | `src/cortex_mem` | 엔진 포트 계약, 런타임, 저장소(PostgreSQL, SQLite, 인메모리), 기준선 엔진, LedgerEngine, 저장소 적합성 테스트 | 없음 (`sqlite3`는 표준 라이브러리). PostgreSQL은 선택: `cortex-mem[postgres]` |
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

## LedgerEngine (증거 원장 엔진)

기록을 주장(claim)으로 묶고, 주장마다 독립 출처별 증거를 쌓고, 게이트를 넘은 주장만 지식으로 올리고, 검증된 실패가 오면 근거와 함께 내린다. 팀 범위에서는 독립 출처를 사람 단위로 세므로 한 사람의 반복은 한 표다. 개인 범위의 주장은 본인만 보므로 세션을 독립 출처로 센다. 같은 사람이 세션 두 개에서 같은 기록을 남기면 표가 둘이 되어, 검증 없이도 개인 지식이 될 수 있다. 팀 지식은 기본으로 검증된 지지(리뷰·테스트·문서·사람 확인)가 하나 이상 있어야 한다. 한 주장에 묶인 기록은 같은 주제라는 것만 알 뿐 같은 말인지는 모르기 때문이다(아래 한계). 건네받은 기억을 다시 말한 기록과 다른 사람의 글을 그대로 옮긴 기록은 표로 세지 않는다. 지식을 받아서 한 일이 성공한 것은 쓸모의 기록일 뿐 참의 표가 아니다. 판정 기록은 `ctx.state`에 두고, 같은 이벤트 순서와 같은 core 묶음이면 같은 결과가 나온다.

엔진 경로는 `cortex_mem.engines.ledger:LedgerEngine`이다. 생성자는 `Mapping[str, str]` 설정만 받고(없으면 기본값), 키가 틀리거나 값이 범위를 벗어나면 `ValueError`로 멈춘다. 서버의 `CORTEX_ENGINE=cortex_mem.engines.ledger:LedgerEngine`은 팩토리를 빈 설정(`{}`)으로 부르므로 기본값만 쓰인다. 설정을 바꾸려면 코드에서 `LedgerEngine({...})`로 만든다.

`ctx.similar`(임베딩을 켰을 때)가 있으면 core 묶음으로 같은 주장을 찾고, 없으면 글자 비교(토큰 자카드)로 대신한다. 토큰은 문자 체계와 상관없이 글자·숫자 묶음이고, 띄어 쓰지 않는 문자(한글, 한자, 가나)는 묶음 뒤에 글자 2-gram을 더한다. 토큰이 하나도 없는 기록(이모지·구두점뿐)은 서명을 만들지 않는다. 글자 비교 경로는 검색 결과를 `match_pool`의 16배까지만 넓힌다.

### 동작

**평가 시점.** 따로 도는 배치는 없다. 런타임이 이벤트를 전달할 때(`append` 직후. 서버는 도구 호출마다 `Runtime.tick`으로 다른 세션이 쌓은 이벤트를 먼저 전달한다) 그 이벤트가 닿는 주장만 다시 평가한다.

| 이벤트 | 하는 일 |
|---|---|
| `record` | 같은 주장을 찾거나 새 주장을 열고 지지 증거를 붙인 뒤 그 주장을 다시 평가한다 |
| `outcome`·`human_confirm` | `refs`가 가리키는 주장마다 지지·반박 증거를 붙이고 다시 평가한다. 검증된 판정(리뷰·테스트·문서·사람 확인)은 주장 id나 그 주장의 대표 기록을 가리킬 때만 센다(아래 한계 첫 줄). 자기 보고 성공은 그 세션이 주장을 건네받은 뒤의 것만 쓸모(`use_success`)로 센다 |
| `recall` | 세션이 건네받은 주장을 표시한다. 평가하지 않는다 |
| `tick` | `review_after_days`가 켜져 있으면 마지막 지지가 오래된 established 주장을 보관한다 |

다시 평가는 증거를 세고(S, C, 독립 출처 수), 대표 기록을 고르고, 게이트를 돌려 다음 상태를 정한 뒤, 상태나 대표 기록이 바뀌었으면 결정을 낸다. 같은 이벤트를 다시 받아도 두 번 세지 않는다.

**상태.** `candidate` → `provisional`(출처 충족) → `established`(게이트 전부 통과) ↔ `disputed`(반박 비율 초과). 검증된 채널의 실패가 있고 C ≥ S이면 `deprecated`, `tick`이 오래 지지 없는 established를 `archived`로 보낸다.

- established·disputed는 반박(서로 오감), 강한 반박(deprecated), tick(archived)으로만 바뀐다. 출처·간격 같은 다른 조건이 약해져도 아래 단계로 내려가지 않는다.
- deprecated·archived는 증거만 쌓는다. 내려간 시각 뒤의 검증된 지지가 와야 다시 게이트를 돈다.
- 게이트의 기준과 정확히 같은 값(반박 비율이 상한과 같음, C = S)은 기준에 닿은 것으로 본다. S와 C는 소수 가중치의 float 합이라 1e-9 허용 오차로 비교한다.
- 대표 기록(지식 본문)은 지금 대표가 가장 강한 지지로 남아 있는 한 더 강한 지지 기록이 올 때만 바뀌고, 승격된 주장의 새 본문은 게이트를 다시 통과해야 한다.

**결정.** established가 되거나 승격된 주장의 대표 기록이 바뀌면 `Promote`(지식 id = 주장 id, 본문 = 대표 기록, 근거 = 지지 이벤트)를 낸다. 승격된 주장이 deprecated가 되면 `Demote(why="invalid")`(근거 = 검증된 반박 이벤트와 마지막 실패), archived가 되면 `Demote(why="stale")`를 낸다. disputed는 지식을 내리지 않는다.

**건넬 때(`handoff`).** 지식 검색·기록 검색(임베딩을 켜면 의미 검색 두 목록도)을 RRF로 합쳐 주장 단위로 모으고, 상태 가중·최신성을 곱해 순위를 낸다. deprecated·archived와 조건이 호출자의 `context`와 다른 주장은 뺀다. 항목 하나가 주장 하나이고, 본문 아래에 `[established · human 1, test 1 · 2 sources · since 2026-10-03]` 같은 출처 줄을 붙이며, `sources`에는 지지 이벤트가 모두 들어간다. `status`는 승격된 established만 `knowledge`, 나머지는 `memory`다.

**상태와 다시 만들기.** 판정 기록은 모두 `ctx.state`에 있다. `state_version`은 상태 형식(`STATE_FORMAT = "2"`: 증거 줄이 기록에서 왔는지(`record`)를 함께 저장한다)과 설정의 해시를 합친 값이다(`2+<해시>`). 설정이 무엇이 지식이 되는지를 정하므로, 다른 설정으로 쌓인 저장소를 이어 쓰지 않고 처음부터 다시 받는다. 같은 값은 쓰는 방법이 달라도(`0.8`과 `0.80`) 같은 해시다. 버전이나 색인 지문이 바뀌면 런타임이 상태를 비우고 처음부터 다시 전달하고, 같은 지식 베이스가 다시 만들어진다. 주장을 만들지 않고 넘긴 이벤트 수는 `ledger:skipped` 키에 이유별로 남는다(`org_scope`, `unknown_outcome`, `unknown_ref`, `out_of_scope_evidence`, `verified_not_canonical`). 같은 이벤트가 다시 전달돼도 한 번만 센다.

설정 키(값은 모두 문자열이다. 기본값은 `config.py`와 같다):

| 키 | 기본값 | 뜻 |
|---|---|---|
| `similar.same_threshold` | `0.0` | core 묶음 안에서 같은 주장으로 보는 이웃 점수(0~1). 0이면 묶음을 그대로 믿는다 |
| `same_threshold` | `0.8` | 글자 비교 대체 경로에서 같은 주장으로 보는 토큰 자카드(0 초과 1 이하) |
| `copy_threshold` | `0.9` | 글자를 옮긴 것으로 보는 토큰 자카드. `same_threshold` 이상 1 이하 |
| `match_pool` | `10` | 같은 주장을 찾을 때 보는 이웃·후보 수(1 이상) |
| `pool` | `50` | `handoff` 후보 수. 후보 목록마다(1 이상) |
| `gate.<scope>.min_sources` | `2` | 지지하는 독립 출처 수. `<scope>`는 `personal`·`team` |
| `gate.<scope>.alt_verified` | `1` | 출처가 모자랄 때 대신 채우는 검증된 지지 출처 수 |
| `gate.<scope>.require_verified` | `false`(personal), `true`(team) | 검증된 지지가 반드시 있어야 하는가(`true`·`false`만). team의 기본값이 `true`인 이유는 아래 한계 첫 줄 |
| `gate.<scope>.min_span_days` | `0` | 첫 지지와 마지막 지지의 최소 간격(일) |
| `gate.<scope>.max_contra_ratio` | `0.20`(personal), `0.15`(team) | 반박 점수 비율 C/(S+C)의 상한(0~1) |
| `weight.<channel>` | observation `1.0`, review `1.5`, test `2.0`, doc `2.0`, human `3.0` | 채널별 증거 가중. `<channel>`은 `observation`·`review`·`test`·`doc`·`human` |
| `trust.observation` | `0.7` | 에이전트 자기 보고(`observation`)의 신뢰 |
| `alpha`, `beta` | `1.0`, `1.0` | 신뢰도 `(S+α)/(S+C+α+β)`의 사전값 |
| `status_weight.<status>` | candidate `0.5`, provisional `0.6`, established `1.0`, disputed `0.4` | 건넬 때 순위에 곱하는 상태 가중 |
| `own_candidate_weight` | `0.7` | 호출자 본인의 candidate 가중 |
| `review_after_days` | `0` | 이 기간 지지가 없는 established를 보관 처리한다. 0이면 끈다 |
| `half_life_days` | `0` | 건넬 때 최신성 가중의 반감기. 0이면 끈다 |
| `high_risk_terms` | 빈 값 | 쉼표로 나눈 단어(공백 제거, 소문자). 본문에 들어 있으면 사람 확인이 필요하다. 바로 앞이 띄어 쓰는 문자의 글자일 때만 빼고(`key`는 `monkey`에 걸리지 않음) 모두 걸린다. 확인을 놓치지 않는 쪽으로, 뒤에는 무엇이 붙어도 되고 앞의 숫자·`_`·띄어 쓰지 않는 문자도 막지 않는다(`passwords`, `DB_PASSWORD`, `새비밀번호`는 걸림) |

가중, 신뢰, `alpha`, `beta`, 상태 가중, `own_candidate_weight`, `*_days`, `min_sources`, `alt_verified`는 0 이상이어야 한다. `alpha`와 `beta`가 둘 다 0일 수는 없다(지지와 반박이 모두 0이면 신뢰도가 0/0이 된다).

한계:

- core 묶음(임베딩이 없으면 글자 비교)은 경쟁하는 사실도 한 주장으로 묶을 수 있다. 조건이 다른 기록과 내려간 주장에 붙는 기록만 나눈다. 묶인 기록은 내용과 상관없이 지지 표가 되므로, 팀 범위는 기본으로 검증된 지지가 있어야 승격한다(`gate.team.require_verified`). 이것을 끄면 값만 다른 기록(포트 8080과 9090 같은)이 두 번째 표가 되어 먼저 쓴 본문이 지식이 될 수 있다. 같은 이유로, 검증된 판정은 주장 id나 그 주장의 대표 기록(지식 본문이 된 기록)을 가리킬 때만 센다. 다른 기록(9090)을 검증한 것은 대표 본문(8080)에 대해 아무것도 말하지 않으므로, 지지로도 반박으로도 세지 않고 `verified_not_canonical`로 남긴다. 이 두 규칙은 **LLM 판정이 도입되기 전까지의 임시 규칙**이다. 묶음 안의 같음·갱신·모순 판정은 작은 LLM 판정이 들어온 뒤에 하고, 그때 이 규칙을 그 판정으로 바꾼다.
- MCP 서버는 기록과 결과를 모두 `signal="self"`로 저장하고 `human_confirm` 도구가 없다. 그래서 서버로 이 엔진을 쓰면 검증된 지지가 들어올 길이 없어, 기본 설정에서는 팀 범위 주장이 지식이 되지 않는다(여러 사람이 기록하고 recall해서 성공을 보고해도 `provisional`에 머문다). 개인 범위는 세션 수로 승격된다. v0.1에서는 의도한 기본값이다.
- 임베딩이 없으면 서로 다른 사람의 기록을 글자만으로는 거의 묶지 못한다.
- v0.1 범위는 personal·team이고 org 범위 기록은 주장을 만들지 않는다.
- 고위험 단어가 든 본문은 그 본문이 대표가 된 뒤의 사람 확인이 있어야 지식이 된다.
- 글자 비교 경로는 이미 받은 기록만 후보로 세지만, BM25 점수 통계는 로그 전체에서 나온다. 그래서 처음부터 다시 만들 때 드물게 다른 주장으로 묶일 수 있다. `ctx.events`가 받은 위치까지만 읽게 되면(#19) 없어진다.
- 보관에서 되살아난 주장이 게이트를 못 넘으면 provisional·candidate로 남고, `tick`은 established만 보관하므로 다시 보관되지 않는다.

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
- 이벤트 하나를 전달하는 데 쓰기 트랜잭션이 6개 든다(기록 추가, 이벤트 본문 캐시, 색인 위치·벡터·묶음, 전달 커밋, 지식 본문 캐시, 지식 벡터). 엔진 실행과 임베딩을 한 트랜잭션에 넣던 이전 구조(4개)보다 많지만 각각 짧고, 느린 엔진이 있어도 다른 세션의 기록이 몇 밀리초 안에 저장된다(엔진이 3초 멈춘 동안 다른 연결의 기록 저장 대기: 2.7초 → SQLite 2ms, PG 12ms.)
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
