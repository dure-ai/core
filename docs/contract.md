# 엔진 포트 계약 v0.3.0

> 상태: 초안 (PR #1 리뷰 2차 반영, D1~D7 합의). 코드 정본은 `src/cortex_mem/contract.py`(포트), `src/cortex_mem/access.py`(scope 규칙), `src/cortex_mem/runtime.py`(런타임).
> 이 문서의 규칙 중 타입으로 표현되지 않는 것(호출 순서, id·scope 의미, 런타임 동작)도 계약의 일부다.

## 1. 무엇이 고정이고 무엇이 교체되나

```mermaid
flowchart TB
  Agent["에이전트"] -->|MCP| Server
  Harness["하니스"] -->|같은 런타임 사용| Runtime
  subgraph Fixed["고정 부분"]
    Server["MCP 서버<br/>도구 인자 → Caller"] --> Runtime["런타임<br/>이벤트 전달 · 결정 검증 · handoff 필터"]
    subgraph Shared["공용 저장소"]
      Log[("이벤트 로그")]
      Wiki[("지식 베이스 (Git 위키)<br/>Knowledge · Skill")]
    end
    State[("엔진 상태 저장소")]
    Runtime -->|append · 검증 통과한 결정 반영| Shared
  end
  Runtime <-->|observe · tick · handoff| Engine["엔진 A / B / C / 기준선"]
  Engine -.->|EngineContext로 읽기| Shared
  Engine -.->|읽기 · 쓰기| State
```

- 각자 만드는 건 `Engine` 하나. 엔진은 `EngineFactory = Callable[[Mapping[str, str]], Engine]`로 만들어진다. 생성자는 엔진 설정만 받고, 저장소는 매 호출의 `ctx`로 받는다.
- 엔진은 공용 저장소를 **읽기만** 하고, 승격·강등은 **결정(Decision)으로 반환**한다. 지식 베이스에 쓰는 건 런타임뿐이다.
- 엔진이 자기 데이터를 남길 곳은 `ctx.state`(엔진 전용 영속 저장소)다.
- 서버와 하니스는 같은 `Runtime`을 쓴다. 그래서 하니스 점수가 서버에서의 동작을 대표한다.

공용 저장소를 엔진에 넘기지 않는 이유: 같은 원본·같은 검색 위에서 승격 로직만 달라야 엔진 간 점수 차이가 승격 로직의 차이가 되고, 1등 엔진에 다른 엔진의 아이디어를 옮겨 붙일 수 있다.

**결정 (통제 변인)**: 검색(`EventReader.search`, `KnowledgeReader.search`)은 공용으로 고정한다. 엔진은 검색 결과를 재정렬하거나 골라낼 수 있지만, 자기 검색 색인을 따로 두지 않는다. 유사도 검색(`SimilarityReader`: 의미 검색과 묶기)도 같은 이유로 공용으로 고정한다. 엔진은 결과를 재정렬하거나 골라낼 수 있지만 자기 벡터 색인을 두지 않는다. 공용 검색을 쓰지 않는 외부 시스템(§8 어댑터)의 결과는 하니스 리포트에서 따로 해석한다.

## 2. 타입

### Event — 관찰 하나

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str | 호출자가 정한다(서버: uuid, 하니스: 시드에서 결정적으로). 중복이면 `append`가 `ValueError` |
| `kind` | `record` · `recall` · `outcome` · `human_confirm` · `commit` | |
| `at` | datetime | **timezone 필수**. 없으면 생성 시 `ValueError` |
| `user`, `team`, `session` | str | 매 호출마다 받는다. 독립 출처를 세는 기본 단위 |
| `scope` | `personal` · `team` · `org` | 이 기록을 누구에게 보일지 (§3-3) |
| `text` | str | record: 기록 내용, recall: 질의, commit: 커밋 메시지 |
| `agent` | str? | 클라이언트 종류 (claude-code, cursor, codex …) |
| `signal` | `self` · `test` · `review` · `human` · `doc`? | 증거 신호 종류 |
| `outcome` | `success` · `failure` · `unknown`? | outcome 이벤트의 결과 |
| `conditions` | Mapping[str, str] | **이 주장이 참인 조건만**. 예: `{"repo": "api", "service": "A"}`. 생성 시 읽기 전용으로 고정된다 |
| `refs` | tuple[str, ...] | 다른 항목을 가리키는 id (§3-2) |
| `repo`, `rev` | str? | 근거가 나온 곳(저장소, 커밋·CI 실행 id). `conditions`와 섞지 않는다 |
| `text_redacted` | bool | 호출자가 받은 `text`를 저장하지 않았다는 표시(기본 `False`). 서버는 저장소 정책보다 넓게 공개되는 outcome에서 판정만 남기고 본문을 비운다(§3-8). 엔진은 "원래 본문이 없는 판정"과 구분할 수 있다 |

`conditions`에는 조건부 사실을 판정하는 데 필요한 맥락(`repo`, `service`, `db` …)을 넣는다. 키 이름은 소문자 snake_case로 쓴다.

### Caller — 지금 부르는 쪽

`user`, `team`, `session`, `at`(timezone 필수), `agent`, `repo`, `context`. 서버는 연결과 설정에서, 하니스는 시나리오(가상 시각)에서 채운다. 질의 범위 `scope`는 Caller와 별도 인자로 받는다.

- `repo`: 지금 작업 중인 저장소. 서버는 프로젝트별 설정 `CORTEX_REPO`를 기본값으로 쓰고, 도구 인자로 덮어쓸 수 있다. 하니스는 문항의 repo. 도구 인자는 위치만 바꾸고 scope 정책은 바꾸지 못한다(§3-8).
- `context`: 질문한 맥락(`{"service": "api"}` 등). 키 규칙은 `conditions`와 같다. 생성 시 읽기 전용으로 고정된다.
- 조건부 사실(`Event.conditions`)이 이 질문에 해당하는지 판단하는 건 **엔진 몫**이다. 런타임은 `repo`·`context`로 거르지 않는다.
- recall 이벤트는 `repo`·`conditions`에 질문한 곳(`Caller.repo`, `Caller.context`)을 남긴다.

### Knowledge — 승격된 항목 하나 (위키 문서 하나)

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str | 엔진이 정한다. **안정적**이어야 한다(위키 경로가 된다) |
| `kind` | `knowledge` · `skill` | |
| `title`, `body` | str | body는 마크다운 |
| `scope` | `personal` · `team` · `org` | 누구에게 보일지 |
| `owner` | str | personal: user id, team: team id, org: `"*"` (v0.1은 조직 하나) |
| `sources` | tuple[str, ...] | 근거 이벤트 id. **비면 거부**된다 |
| `conditions` | Mapping[str, str] | 이 항목이 유효한 조건 |

### Decision — 엔진이 반환하는 결정

- `Promote(item)`: 항목을 만들거나, 같은 id의 항목을 바꾼다.
- `Demote(id, reason, sources, why="invalid", since=None)`: 항목을 지식 베이스에서 뺀다. `sources`는 강등 근거(실패 outcome 등).
  - `why`: `invalid`(틀렸거나 틀리게 됨) · `stale`(더 이상 쓸모없음, 보관). 런타임 처리는 같고 하니스가 따로 센다.
  - `since`: 언제부터 참이 아닌지(알면, timezone 필수). 위키는 Git이라 내린 항목도 이력에 남으므로, 유효 기간은 "Promote 커밋 ~ `since`"로 보인다.

결정 종류는 "런타임이 다르게 처리해야 하는가"를 기준으로 두 가지만 둔다.

| 하려는 일 | 결정 |
|---|---|
| 승격 | `Promote(item)` |
| 갱신 | 같은 id로 `Promote(item)` |
| Skill화 | `Promote(item)`, `item.kind = "skill"` |
| 무효화 | `Demote(id, ..., why="invalid", since=...)` |
| 보관(오래된 기억 정리) | `Demote(id, ..., why="stale")` |

위키에 남긴 채 검색에서만 빼는 것처럼 처리가 달라져야 하는 종류가 필요해지면 그때 추가한다.

### Handoff — 엔진이 에이전트 컨텍스트에 넣는 항목 하나

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str | **호출 사이에 안정적**이어야 한다(recall 이벤트의 `refs`로 돌아온다). 지식 항목이면 그 id |
| `content` | str | 에이전트에게 보일 내용 |
| `sources` | tuple[str, ...] | 근거 이벤트 id. 호출자가 볼 수 없는 것이 하나라도 있으면 런타임이 뺀다 (§3-5) |
| `status` | `memory` · `knowledge` · `invalidated`? | **선택**. 에이전트 표시용이며 채점에는 쓰지 않는다 |
| `conditions` | Mapping[str, str] | 이 항목이 유효한 조건 |
| `confidence` | float? | 0..1. 주면 Brier 점수로 확률 채점 |

### 엔진이 받는 것과 구현하는 것

```python
class EventReader(Protocol):            # 이벤트 로그, 읽기 전용
    def get(self, ids) -> list[Event]: ...
    def search(self, query, caller, scope, k=20, kinds=None) -> list[tuple[Event, float]]: ...

class KnowledgeReader(Protocol):        # 지식 베이스의 현재 상태, 읽기 전용
    def get(self, ids) -> list[Knowledge]: ...
    def all(self) -> list[Knowledge]: ...
    def search(self, query, caller, scope, k=20) -> list[tuple[Knowledge, float]]: ...

class StateStore(Protocol):             # 엔진 전용 영속 저장소, JSON 값
    def get(self, key): ...
    def put(self, key, value): ...
    def delete(self, key): ...
    def scan(self, prefix=""): ...

@dataclass(frozen=True)
class EngineContext:
    events: EventReader
    knowledge: KnowledgeReader
    state: StateStore
    similar: SimilarityReader | None = None   # 임베딩이 설정되지 않으면 None. 엔진이 그 경우를 처리해야 한다

@dataclass(frozen=True)
class Group:                            # 같은 (scope, owner) 안에서 비슷하다고 묶인 record들
    id: str                             # 첫 멤버의 이벤트 id
    scope: Scope
    owner: str                          # access.owner_for와 같은 규칙
    members: tuple[str, ...]            # record 이벤트 id, append 순

class SimilarityReader(Protocol):       # 코어가 만든 임베딩 색인, 읽기 전용
    model: str                          # 색인을 만든 임베딩 모델(정보용)
    def search(self, query, caller, scope, k=20, kinds=None) -> list[tuple[Event, float]]: ...
    def neighbors(self, event_id, caller, scope, k=20) -> list[tuple[Event, float]]: ...
    def knowledge(self, query, caller, scope, k=20) -> list[tuple[Knowledge, float]]: ...
    def group(self, event_id) -> Group | None: ...

class Engine(Protocol):
    state_version: str                  # 선택. 바뀌면 상태를 버리고 처음부터 다시 전달 (§3-8)
    def observe(self, event, ctx) -> Sequence[Decision]: ...
    def tick(self, now, ctx) -> Sequence[Decision]: ...
    def handoff(self, query, caller, scope, ctx, k=6) -> list[Handoff]: ...
```

- **점수**: `search`·`neighbors`·`knowledge`의 점수는 코사인 유사도(−1..1)이고 "높을수록 가깝다". 텍스트 검색 점수와 척도가 다르므로 섞지 않는다. 벡터는 float32로 저장되므로 같은 데이터라도 저장소(인메모리·SQLite·PostgreSQL)에 따라 점수가 마지막 자리(1e-6 안)에서 다를 수 있다. 같은 저장소에서는 실행마다 같다.
- **가시성**: 세 검색 모두 `access.event_visible`·`knowledge_visible`로 거른 집합 안에서만 찾는다(SQL 저장소는 같은 규칙의 SQL 조건으로 거른다, §5). `group()`은 호출자를 받지 않는다. 엔진은 모든 이벤트를 관찰하므로 묶음 정보가 엔진에게 새 노출이 아니고, 엔진이 내보내는 것은 결정 검증과 handoff 필터를 지난다(§3-5).
- **묶음 멤버를 handoff에 쓰면 근거로 인용한다**: handoff 필터는 Handoff가 `sources`에 적은 이벤트만 검사한다. 엔진이 `group().members`(또는 `neighbors` 결과) 이벤트의 내용을 Handoff의 `content`에 넣으면 그 이벤트들을 모두 `sources`에 넣어야 한다. 빠뜨리면 호출자가 볼 수 없는 기록의 내용이 필터를 지나 나간다.
- **엔진이 본 데까지만**: `similar`는 엔진이 이미 받은 이벤트까지만 읽는다. `observe` 중에는 지금 관찰하는 이벤트까지, `tick`·`handoff` 중에는 엔진의 커서까지다. 그 뒤의 이벤트는 `search`·`neighbors` 결과에도, `group().members`에도 나오지 않고, 그 뒤의 이벤트로 `group()`을 부르면 `None`이다. 그래서 처음부터 다시 전달할 때(§3-8)도 처음 전달 때와 같은 결과를 읽는다(§3-7). 지식(`knowledge`)은 지금의 지식 베이스다.
- **recall 질의**: `search`·`knowledge`의 질의는 임베딩하되 저장하지 않는다(프로세스 메모리의 작은 LRU만).

`Engine`은 `runtime_checkable`이라 하니스가 `isinstance`로 구현 여부를 확인한다. 위키를 검색해 건네는 기본 handoff는 `cortex_mem.kit.handoff_from_knowledge`로 제공한다(선택).

## 3. 규칙

### 3-1. 런타임과 호출 순서

1. 모든 이벤트는 `append` 순서대로 **정확히 한 번** `observe`에 전달된다. 커서는 런타임이 관리하고, 엔진은 재생이나 커서를 신경 쓰지 않는다.
2. 이벤트 하나의 전달은 트랜잭션 하나다. `observe`가 반환하면 그 호출의 상태 쓰기, 결정 반영, 커서 이동이 함께 반영된다. `observe`가 예외를 던지면 아무것도 반영되지 않고, 그 이벤트는 다음 `sync` 때 다시 전달된다. 다른 프로세스가 먼저 전달했으면(`CursorConflict`) 이쪽 결과는 버리고 커서를 다시 읽는다.
   - **계속 실패하는 이벤트**: `Runtime(..., max_failures=N)`이면 같은 이벤트에서 `observe`가 N번 실패했을 때 그 이벤트를 건너뛴다. 실패한 전달 트랜잭션은 롤백하고(PostgreSQL에서 DB 오류로 중단된 트랜잭션도), 새 트랜잭션에서 커서만 넘긴다. 그 호출의 쓰기는 남기지 않으며, 그 커밋이 성공한 뒤에만 `Runtime.skipped`(`seq`, `event_id`, `attempts`, `error`)에 남긴다(커밋이 실패하면 그 오류를 올리고 다음 시도에서 다시 건너뛴다). 이벤트 자체는 로그에 남는다. 기본값 `None`은 건너뛰지 않고 매번 예외를 던진다.
   - 하니스는 `None`으로 쓴다. 엔진 실패를 조용히 넘기면 "엔진이 실패했는데 점수는 나온" 상태가 되기 때문이다. 서버는 한 이벤트가 모든 세션을 막지 않도록 `CORTEX_MAX_FAILURES`(기본 3)로 켠다.
   - 실패 횟수와 `skipped`는 `verdicts`처럼 프로세스 메모리에만 있다(영속화는 §6).
   - **색인 뒤 전달**: 임베딩이 설정되어 있으면 이벤트는 색인된 뒤 전달된다. 색인 실패는 그 이벤트를 전달하지 않고 예외를 올린다. `max_failures`로 건너뛰지 않는다.
3. 이벤트는 `append` 뒤에 전달되므로, 엔진은 `observe` 안에서 그 이벤트를 `ctx.events.get`할 수 있다.
4. `handoff` 직전에 런타임이 밀린 이벤트를 먼저 전달한다. 그래서 세션 B도 세션 A가 남긴 기록을 반영한 상태로 답한다.
5. recall 한 번은 이렇게 흐른다(`Runtime.recall`).
   ```
   handoffs = runtime.handoff(query, caller, scope, k)        # 필터 통과분
   runtime.append(Event(kind="recall", text=query, refs=tuple(h.id for h in handoffs), ...))
   return handoffs  # 에이전트에게
   ```
   엔진은 자기가 건넨 것이 무엇인지 recall 이벤트로 다시 관찰한다. recall 이벤트는 검색한 범위와 상관없이 `personal`로 남는다. 질의는 그 사람이 무엇을 하는지 드러내기 때문이다.

### 3-2. `refs` 의미

| kind | refs |
|---|---|
| `recall` | 에이전트에게 건넨 Handoff id |
| `outcome`, `human_confirm` | 판정 대상인 이벤트 또는 Handoff id. **비면 `Runtime.append`가 `ValueError`로 거부한다.** 가리키는 id가 있는지는 검사하지 않는다(Handoff id는 지식 항목이 아닐 수 있고, 이미 강등됐을 수 있다) |
| `record` | 이 기록을 쓰기 직전에 건네받은 Handoff id (알면). 에이전트가 받은 걸 다시 말한 건지 구분하는 단서 |
| `commit` | 비워 둔다 |

### 3-3. scope 가시성 (`access.py`)

범위는 personal < team < org 순으로 넓어진다.

| 질의 scope | 보이는 이벤트 | 보이는 지식 |
|---|---|---|
| `personal` | 내 이벤트 | 내 personal 항목 |
| `team` | + 우리 팀의 `team` 이벤트 | + 우리 팀의 team 항목 |
| `org` | + 모든 `org` 이벤트 | + org 항목 |

다른 사람의 `personal` 이벤트와 다른 팀의 `team` 이벤트는 어떤 scope로도 보이지 않는다.

### 3-4. 결정 검증

런타임은 결정을 지식 베이스에 반영하기 전에 검사한다. 통과한 결정은 **사람 승인 없이 즉시** 반영된다. 모든 결정은 `Runtime.verdicts`에 `(결정, 승인 여부, 거부 사유, 시각)`으로 남고(`rejections`는 거부된 것만), 하니스는 이를 **관찰 지표**로 리포트에 싣는다. 판정 지표에는 넣지 않는다(하니스는 에이전트에게 닿은 결과로만 채점).

- `Promote`: id가 비어 있지 않고, `sources`가 비어 있지 않고, 모두 존재하고, **모두 그 항목을 볼 사람 전원에게 보여야** 한다.
- `Demote`: 대상 항목이 있어야 하고, `sources`가 위와 같은 조건을 그 항목의 범위에 대해 만족해야 한다.
- 한 번에 반환한 결정들은 순서대로 검사된다. 앞에서 승격한 항목을 뒤에서 강등할 수 있다.

| 항목 scope | 근거로 쓸 수 있는 이벤트 |
|---|---|
| personal (owner = user) | 그 사용자의 이벤트, org 이벤트 |
| team (owner = team) | 그 팀의 `team` 이벤트, org 이벤트 |
| org | org 이벤트 |

그래서 **personal 기록은 팀 지식의 근거가 될 수 없다.** 서로 다른 사람 여러 명의 personal 기록이어도 같다. 사람 사이의 교차 확인으로 승격하는 경로는 기록이 처음부터 team으로 남아야 열린다. 이 기본값은 에이전트가 아니라 서버 설정이 정한다(§3-8).

personal 기록 여러 개로 팀 지식을 만드는 것(롤업)을 v0.1에서 허용하지 않는 이유:
- 허용해도 handoff 필터(§3-5)가 팀원에게는 그 항목을 빼게 된다. 팀원은 근거를 볼 수 없기 때문이다. 필터까지 풀면 "근거는 모두 호출자가 볼 수 있어야 한다"(하니스 M3)가 깨진다.
- 개인 식별 정보 검사는 문자열 검사라, 엔진이 요약하거나 바꿔 쓰면 개인 내용이 통과한다. 구조로 보장할 수 없다.
- v0.2 후보: 런타임이 "서로 다른 N명이 같은 주장을 함"이라는 집계 이벤트를 만들고(원본 id 비공개), 그것을 근거로 쓰는 방식.

### 3-5. handoff 필터

런타임은 엔진이 건넨 항목 중 다음을 뺀다: `sources`가 비었거나, 없는 이벤트를 가리키거나, 호출자가 그 scope로 볼 수 없는 이벤트가 하나라도 있는 항목. 항목 id가 지식 항목이면 그 항목도 호출자에게 보여야 한다. 그 뒤 `k`개로 자른다. 필터는 `sources`에 적힌 이벤트만 보므로, `content`에 쓴 이벤트(묶음 멤버 등)는 모두 `sources`에 적어야 한다(§2).

누출을 막는 지점은 결정 검증(§3-4)과 이 필터 두 곳뿐이고, 둘 다 고정 부분에 있다. 엔진은 모든 이벤트를 관찰하지만, 나가는 것은 항상 검사된다.

### 3-6. 시간

엔진은 시계를 직접 읽지 않는다. `event.at`, `tick(now)`의 `now`, `caller.at`만 쓴다. 서버는 이벤트를 전달할 때 현재 시각으로, 하니스는 시나리오가 정한 시각에 `tick`을 부른다. 오래된 기억 정리와 만료는 `tick`에서 결정으로 반환한다.

### 3-7. 결정성

같은 이벤트·tick 순서를 넣으면 같은 결정과 같은 `handoff`가 나와야 한다. 하니스가 재실행 뒤집힘률(목표 10% 이하)을 잰다. 엔진 안에서 난수를 쓰면 시드를 고정한다. 런타임도 이것을 돕는다: `ctx.similar`는 엔진이 받은 이벤트까지만 읽으므로(§2), 다른 엔진이 이미 뒤의 이벤트까지 색인해 두었든, 상태 버전이 바뀌어 처음부터 다시 받든 같은 이벤트에서 같은 묶음과 이웃을 본다.

### 3-8. 엔진 상태와 scope 기본값

- 오래 남길 상태는 `ctx.state`에만 둔다. 인스턴스 필드는 캐시로만 쓴다. 서버 프로세스가 여러 개 떠서 같은 저장소를 쓸 수 있기 때문이다.
- `ctx.state`의 값은 JSON이고, 넣고 꺼낼 때 복사된다. 커밋 전의 값도 마찬가지다(런타임이 JSON 문자열로 들고 있다가 꺼낼 때마다 새로 만든다). `handoff`에서는 읽기 전용이다.
- 값 하나는 JSON으로 64 KiB까지다(`MAX_STATE_VALUE_BYTES`). 넘으면 `put`이 `ValueError`를 던진다.
- **권장: 대상 하나당 키 하나.** 상태 전체를 값 하나에 넣으면 이벤트마다 전체를 다시 쓰게 된다. 예시:
  ```
  memory:<memory_id>                  L1 기억 하나
  claim:<claim_id>                    L2 주장 하나 (정규화 키, 상태, 조건, 점수)
  claim_key:<canonical_key>           정규화 키 → claim_id (중복 확인용 색인)
  evidence:<claim_id>:<event_id>      증거 한 줄 (지지/반박, 독립성 키, 종류)
  retrieval:<session>:<handoff_id>    이 세션에 건넨 항목 (자기강화 차단용)
  ```
  증거 키에 `event_id`를 넣으면 같은 이벤트를 다시 받아도 두 번 세지 않는다(재전달에 대한 안전장치).
- **색인 버전**: 색인 버전(임베딩 모델, 묶음 문턱값)이 바뀌면 엔진 상태를 비우고 이벤트 로그 처음부터 다시 전달한다. 엔진의 `state_version`이 바뀐 것과 같은 처리다. 색인은 저장소(파일)에 하나지만 엔진 상태는 엔진마다 있으므로, 런타임은 엔진마다 "이 상태를 어느 색인 버전 위에서 만들었는지"를 그 엔진의 상태에 `_runtime/index` 키로 남기고 `sync` 때마다 확인한다. 같은 저장소의 다른 엔진이 먼저 색인을 다시 만들었어도 이 엔진의 표시가 지금 색인 버전과 다르면 이 엔진도 처음부터 다시 받는다.
  - `_runtime/`로 시작하는 상태 키는 임베딩 설정과 관계없이 런타임 몫이니 엔진은 쓰지 않는다. 엔진의 `ctx.state`에서는 보이지 않고(`get`은 `None`, `scan`에서 빠짐), `put`·`delete`는 `ValueError`를 던진다.
  - **주의: 저장소 하나에는 색인 설정 하나.** 임베딩 모델이나 `CORTEX_GROUP_THRESHOLD`가 다른 프로세스가 같은 DB(파일)를 쓰면 서로 자기 설정으로 색인을 다시 만들어, 색인이 왔다 갔다 하고 엔진들이 계속 처음부터 다시 받는다. 설정을 바꿀 때는 그 DB(파일)를 쓰는 프로세스를 모두 내리고 새 설정으로 다시 띄운다.
- **상태 형식 버전**: 엔진에 `state_version: str`을 두면, 저장된 버전과 다를 때 런타임이 상태를 지우고 커서를 0으로 돌려 이벤트 로그 처음부터 다시 전달한다. 마이그레이션 코드는 필요 없다. 지식 베이스는 지우지 않는다(다시 나온 Promote가 같은 id를 덮어쓴다). 버전이 다른 서버 프로세스를 동시에 띄우지 않는다.
- record의 scope 기본값은 서버 설정(저장소 → 기본 scope)이 정하고, 에이전트는 기본값보다 **좁히는 것만** 할 수 있다. 저장소 정책은 설정된 저장소(`CORTEX_REPO`)로만 고른다. 도구 인자 `repo`는 근거 위치(`Event.repo`)만 바꾸므로, 기본값이 더 넓은 저장소 이름을 넘겨 범위를 넓힐 수 없다. 설정된 저장소가 없으면 기본 정책(`CORTEX_RECORD_SCOPE`)만 적용한다. recall의 기본 scope는 볼 수 있는 가장 넓은 범위다.
- outcome(결과 보고)의 scope 기본값은 **판정 대상 지식의 범위**다. 그래야 팀 지식을 강등하는 근거가 된다. 에이전트는 좁히는 것만 할 수 있다. 하니스처럼 scope를 직접 정하는 호출자는 팀 지식에 대한 결과 이벤트를 team으로 넣어야 강등이 일어난다.
- 그 범위가 설정된 저장소의 정책보다 넓으면(예: personal 정책 저장소에서 org 지식을 판정) 서버는 판정(`outcome`·`signal`·`refs`)만 남기고 `text`를 비운 뒤 `text_redacted=True`로 표시한다. 강등에 필요한 건 판정이고, 에이전트가 쓴 본문(경로, 내부 사정)은 저장소 정책을 따른다. refs에 범위가 다른 항목이 섞이면 가장 넓은 범위 하나로 남는다.

## 4. 하니스

### 하니스 이벤트 ↔ Event 대응

| 하니스 이벤트 | kind | 주요 필드 |
|---|---|---|
| 에이전트 기록 | `record` | `text`, `agent`, `signal="self"`, `conditions` |
| recall 요청과 응답 내용 | `recall` | `text`=질의, `refs`=건넨 Handoff id |
| CI·테스트 결과 (앞선 기록·항목을 판정) | `outcome` | `signal="test"`, `outcome`, `refs`(필수), `repo`, `rev` |
| CI·테스트 결과 (단독 증거) | `record` | `signal="test"`, `outcome`, `repo`, `rev` |
| 사람 확인 | `human_confirm` | `signal="human"`, `outcome`, `refs` |
| 문서 근거 | `record` | `signal="doc"`, `repo`, `rev` |
| 대상 코드 커밋(여러 저장소) | `commit` | `repo`, `rev`, `text`=커밋 메시지 |

### 채점

- 엔진마다 시나리오별로 새 `InMemoryDatabase`와 `Runtime`을 만들어 같은 이벤트를 흘려 넣는다. 하니스는 엔진을 직접 부르지 않고 런타임만 부른다(서버와 같은 진입점).
- 주 채점은 에이전트에게 건넨 결과(handoff)로 한다. 지식 베이스 상태(`knowledge.all()`)는 신뢰도 채점(Brier, v0.2) 같은 선택 기능에만 쓴다.
- 판정 기준선은 같은 런타임 위의 `PromoteAllEngine`이다. **비교는 인메모리 저장소 기준**이다. 저장소 구현마다 검색 점수 함수가 달라(인메모리 BM25, SQLite FTS5 `bm25`, PostgreSQL 역색인 BM25) 같은 데이터에서도 순위가 달라질 수 있으므로, 하니스 비교는 저장소 구현 하나로 고정한다. 인메모리 BM25는 하니스 `all_store`와 같은 공식·상수(`k1=1.5`, `b=0.75`)를 써서, 기준선과의 점수 차이가 검색 공식이 아니라 엔진 판단에서 나오게 한다.
- 런타임은 `max_failures=None`(기본값)으로 만든다(§3-1).
- 관찰 지표: 결정 기록(`verdicts`: 승인·거부, 사유, 근거 id), `why`별 강등 수, handoff 필터에 걸린 수.

## 5. 고정 부분의 저장소 인터페이스 (`store/base.py`)

엔진은 보지 않는다. 저장 계층 구현(SQLite, PostgreSQL)이 지킬 것이다.

```python
class EventStore(EventReader, Protocol):
    def append(self, event) -> str: ...
    def since(self, seq) -> Iterator[tuple[int, Event]]: ...   # 위치는 1부터, 커밋 순서대로 늘어나기만

class KnowledgeStore(KnowledgeReader, Protocol):
    def put(self, item) -> None: ...
    def remove(self, id) -> None: ...

class EngineStateBackend(Protocol):                           # 엔진 하나의 상태 + 커서
    def get(self, key): ...
    def scan(self, prefix=""): ...
    def cursor(self) -> int: ...
    def commit(self, expected, cursor, writes, deletes) -> None: ...  # 커서가 expected가 아니면 CursorConflict
    def version(self) -> str | None: ...                      # 상태를 쓴 엔진의 state_version
    def reset(self, version) -> None: ...                     # 키 전부 삭제, 커서 0, 버전 기록

class Database(Protocol):                                     # 한 연결 뒤의 저장소 전부
    events: EventStore
    knowledge: KnowledgeStore
    vectors: VectorBackend                                    # 임베딩을 쓰지 않아도 있어야 한다
    def state(self, engine) -> EngineStateBackend: ...       # 엔진별 이름공간
    def transaction(self): ...

class VectorBackend(Protocol):                                # 색인 상태·벡터·묶음·캐시 + 유사도 계산
    kind: str                                                 # "memory" | "sqlite" | "postgres"
    ...                                                       # meta/reset/indexed, put_vector/vector/vectors, put_group/members/partition, cached/cache
    def nearest_member(self, scope, owner, vector, upto) -> tuple[str, float] | None: ...
    def search_events(self, vector, caller, scope, k, kinds=None, upto=None, exclude=None) -> list[tuple[str, float]]: ...
    def search_knowledge(self, vector, caller, scope, k) -> list[tuple[str, float]]: ...
```

- `cortex_mem.testing`의 `EventStoreContract`, `KnowledgeStoreContract`, `EngineStateBackendContract`, `VectorBackendContract`, `DatabaseContract`를 상속해 `make()`만 구현한 테스트가 통과하면 계약을 지킨 것이다(`tests/test_store.py` 참고). `DatabaseContract`는 런타임을 올려 정확히 한 번 전달, 실패 시 전부 롤백, 재접속 후 유지, 상태 버전 교체를 확인하고, `transaction()` 안의 벡터 쓰기가 예외와 함께 되돌려지는지도 본다(되돌릴 수 없는 단일 프로세스 저장소는 `rolls_back = False`). `search` 점수는 "높을수록 좋음"만 약속한다.
- 영속 구현은 `Runtime(..., transaction=...)`에 쓰기 트랜잭션을 넘겨, 커서·엔진 상태·지식 반영을 **한 트랜잭션**으로 묶고 프로세스 사이에서 직렬화한다. 인메모리 구현은 단일 프로세스라 기본값(`nullcontext`)을 쓴다.
- `Database`에는 `vectors: VectorBackend` 속성이 있다. `Database` 프로토콜이 요구하므로 임베딩을 쓰지 않는 배포에서도 구현해야 한다(쓰지 않으면 비어 있을 뿐이다). `VectorBackend`는 색인 상태(모델, 묶음 문턱값, 색인한 마지막 위치), 이벤트·지식 벡터, 묶음, 임베딩 캐시를 맡는다. 모든 쓰기는 런타임의 `transaction()` 안에서 일어나고 그 트랜잭션과 함께 되돌려진다. 이벤트 하나의 색인과 전달은 같은 트랜잭션에 묶이고, 임베딩 캐시는 그 앞의 짧은 트랜잭션에서 미리 채운다. 벡터는 **float32**다: 쓰기(벡터와 임베딩 캐시)는 모두 float32로 반올림해 저장하고, 읽기는 그 값을 파이썬 float로 돌려준다. NaN·무한대·float32 범위를 넘는 값은 `ValueError`로 거절한다(쓰기와 질의 모두). SQLite는 스키마 5에서 BLOB을 float32로 바꿨고, 스키마 4 파일을 열면 float64 벡터·묶음·색인 메타·임베딩 캐시를 지운다(다음 `sync`가 다시 임베딩한다). 기존 `Database`의 다른 부분은 그대로다. 포트 정의와 구현은 임베딩 설계(`docs/specs/2026-11-15-core-embedding-design.md` §5)를 따른다.
- **유사도 계산은 저장소가 한다** (0.3.0). `Index`는 임베딩·캐시·묶음 배정·커서를 맡고, 코사인 점수는 `VectorBackend`의 세 메서드가 낸다. 질의 벡터도 float32로 반올림해 계산한다.
  - `nearest_member(scope, owner, vector, upto)`: 그 (scope, owner) 칸의 묶인 이벤트 중 위치 ≤ `upto`이고 가장 가까운 것과 그 점수. 점수와 상관없이 가장 가까운 것을 돌려주고(문턱값은 `Index`가 본다), 동점은 위치가 작은 것, 칸이 비었으면 `None`. 언제나 정확 검색이다(근사 색인을 쓰지 않는다).
  - `search_events(vector, caller, scope, k, kinds, upto, exclude)`: 호출자가 그 범위에서 볼 수 있는 묶인 이벤트 중 점수 > 0인 것 k개, 점수 내림차순, 동점은 위치 순. `kinds`는 이벤트 종류, `upto`는 위치 상한(`None` = 상한 없음), `exclude`는 뺄 id 하나.
  - `search_knowledge(vector, caller, scope, k)`: 볼 수 있는 지식 중 점수 > 0인 것 k개, 동점은 id 순.
  - **위치**는 `put_group`에 준 seq(= 이벤트 로그 위치)다. 묶음이 없는 이벤트 벡터는 검색 후보가 아니고, 이벤트·지식 저장소에 없는 id도 후보가 아니다.
  - 가시성은 같은 `Database`의 이벤트·지식으로 판단한다. 그래서 백엔드는 그 옆에서 만든다(인메모리는 `InMemoryVectorBackend(events, knowledge)`, SQL 저장소는 `events`·`knowledge` 표와 잇는다). 파이썬 구현은 `access.event_visible`·`knowledge_visible`, SQL 구현은 텍스트 검색과 같은 조건(`store/text.py`의 `EVENT_VISIBLE_SQL`·`KNOWLEDGE_VISIBLE_SQL`)을 쓴다. `VectorBackendContract`의 `make()`는 `.events`·`.knowledge`·`.vectors`를 가진 `Database`를 돌려주고, 범위 × 호출자 조합의 기대 id 표로 두 구현이 같은지 고정한다.
  - 인메모리와 SQLite는 같은 파이썬 코드로 계산하므로 점수가 비트까지 같다. PostgreSQL(pgvector)은 DB 안에서 계산하므로 점수가 1e-6 안에서 같다고 본다. 색인 버전(`Index.want()`)에 `store: backend.kind`가 들어 있어 저장소를 옮기면 색인을 다시 만들고 엔진에 처음부터 다시 전달한다.
- 커서의 compare-and-set(`expected`)은 여러 프로세스가 같은 엔진 상태를 쓸 때 이중 전달을 막는다.
- **append의 위치는 커밋 순서대로 늘어나야 한다.** 커서는 "이 위치까지 전달했다"는 뜻이라, 작은 위치가 큰 위치보다 늦게 커밋되면 그 이벤트는 영영 전달되지 않는다. SQLite는 쓰기가 한 번에 하나라 자동으로 지켜진다. PostgreSQL의 시퀀스(`bigserial`·`IDENTITY`)는 번호를 먼저 받은 트랜잭션이 나중에 커밋될 수 있으므로, 스키마는 `seq bigint PRIMARY KEY`로 두고 PG 구현(`PostgresDatabase`)은 `transaction()`을 바깥에서만 `BEGIN` + `pg_advisory_xact_lock`(DB 전체 잠금 하나)으로 열고, 위치를 잠금 안에서 `max(seq)+1`로 정한다. 롤백된 append는 번호를 남기지 않는다. 잠금 대기는 `lock_timeout`(서버 `CORTEX_DB_LOCK_TIMEOUT`, 기본 `30s`)으로 끊고 분명한 오류를 낸다. 트랜잭션 안에서 DB 오류를 잡아 넘기면 안 된다(중단된 트랜잭션은 커밋되지 않는다). 동시 append·전달을 섞는 시험은 `tests/test_postgres.py`와 `tests/test_similarity_behaviour.py`에 있다(`DatabaseContract`만으로는 잡히지 않는다).
- **한국어 검색**: 토큰(`words`, 한글 2-gram `grams`)을 Python에서 미리 만들어 저장한다. PostgreSQL은 같은 토큰(`store/text.py`)을 역색인 표(`event_terms`, `knowledge_terms`, 문서 길이 표)에 쓰고 BM25(k1=1.2, b=0.75)를 SQL 집계로 계산한다. 기본 전문 검색(`tsvector`)은 한국어를 나누지 못하고 IDF가 없어 쓰지 않으며, 확장도 필요 없다. df·문서 수·평균 길이는 호출자가 볼 수 있는 기록만으로 계산한다(숨은 기록이 점수를 바꾸지 않는다). 점수 값은 저장소마다 다르고 순서의 성질만 같다.
- **PostgreSQL 저장 형식**: JSON(`conditions`, `refs`, 엔진 상태 값 등)과 시각은 `text`로 저장해 키 순서와 시간대·마이크로초가 그대로 돌아온다(`jsonb`·`timestamptz`를 쓰지 않는다). id로 정렬하거나 동점을 가를 때는 `COLLATE "C"`로 파이썬·SQLite와 같은 바이트 순서를 쓴다. 벡터는 pgvector `vector`(float32)이고 점수는 `1 - (emb <=> q)`다. 정확 검색이 기본이고, 근사 색인(HNSW, `CORTEX_VECTOR_INDEX=hnsw`)은 `search_events`·`search_knowledge`에만 쓰며 기본은 꺼져 있다. 근사 색인은 진짜 최근접을 놓치거나 가시성 필터 때문에 k개보다 적게 낼 수 있다. 차원이 4000을 넘으면 색인 없이 정확 검색만 된다. 한 색인 버전 안에서 벡터 차원이 다르면 쓰기에서 `ValueError`를 낸다.

### 계약 변경 이력

| 버전 | 바뀐 것 | 엔진 영향 |
| --- | --- | --- |
| 0.2.0 | 코어 임베딩 색인과 읽기 전용 `ctx.similar`(`SimilarityReader`: search·neighbors·knowledge·group), `Database.vectors`, `VectorBackend` | 엔진이 `ctx.similar`를 쓸 수 있다(없으면 `None`) |
| 0.3.0 | 벡터는 float32로 저장(기존 float64). 유사도 계산을 `VectorBackend`로 옮김(`nearest_member`, `search_events`, `search_knowledge`, `kind`). 백엔드는 같은 DB의 이벤트·지식을 받아 가시성을 판단. 색인 버전에 저장소 종류(`store`)가 들어가 처음 한 번 색인을 다시 만든다. 적합성 시험 `test_vectors_round_trip_as_float64` → `test_vectors_round_trip_at_float32_precision` | `SimilarityReader`는 그대로. 점수가 float32 반올림만큼(1e-6 안) 달라질 수 있다. PostgreSQL + pgvector 구현(`PostgresDatabase`)이 같은 적합성 시험을 통과하는 기준 저장소로 추가된다 |

## 6. v0.2 이후로 미룬 것

- **LLM 주입**: 엔진이 LLM을 쓰면 하니스가 모델 버전과 호출 예산을 정해 주입한다. `EngineContext`에 필드로 추가하므로 엔진 시그니처는 바뀌지 않는다.
  - 제공자는 Upstage, 예산은 팀 크레딧 합계 $330(1인 $110). 엔진·실행별 예산 배분은 하니스 설계에서 정한다.
  - **기존 연구 채점이 시작되기 전까지 필요하다.**
- **HTTP 전송과 인증**: 토큰 → (user, team). 서버 PR 이후.
- **팀 간(org) 승격**: 여러 팀의 지식을 org로 올리는 경로. v0.1은 org 이벤트만 org 지식의 근거가 된다.
- **결정 기록의 영속화와 엔진에 알리기**: v0.1은 `Runtime.verdicts`와 `Runtime.skipped`(메모리)에만 남는다. 위키 커밋 메시지에 같은 내용을 남기는 것은 위키 반영기와 함께.
- **personal 기록 롤업**: §3-4의 집계 이벤트 방식.
- **사람 승인·커밋 입력 경로**: 관리 CLI, Git·CI 훅으로 `human_confirm`·`commit` 이벤트를 넣는 관례.
- 계약을 바꿀 때는 `CONTRACT_VERSION`을 올리고 근거를 이 문서에 남긴다.

## 7. 팀 확인 필요

**합의됨 (10/20, PR #1·#2 리뷰)**
- [x] D1 저장소의 주인: 고정 부분이 소유, 엔진은 결정만 반환 (§1)
- [x] D2 승격 즉시 반영, 결정 기록은 관찰 지표 (§3-4)
- [x] D3 엔진 포트 모양: `observe` / `tick` / `handoff` + `EngineContext` (§2)
- [x] D4 엔진 상태 저장소: 키-값(JSON), 같은 트랜잭션으로 커밋, 상태 버전 (§3-8)
- [x] D5 scope 정책: 기본값은 서버 설정, 에이전트는 좁히기만, 팀 간 승격은 v0.1 제외 (§3-4, §3-8)
- [x] D6 handoff는 엔진 몫, 범위 필터는 런타임 (§3-5)
- [x] D7 서버 전송: 최종 HTTP 하나, v0.1은 공용 DB + stdio 가능 (PR #2)

**합의됨 (10/20, 2차 리뷰)**
- [x] 결정 종류는 Promote·Demote 두 가지 + `why`·`since` (§2)
- [x] personal 기록 롤업은 v0.1 제외 (§3-4)
- [x] outcome scope 기본값 = 판정 대상 지식의 범위 (§3-8)
- [x] 질문 맥락 `Caller.repo`·`context` (§2), outcome·human_confirm의 refs는 append에서 검사 (§3-2)

**합의됨 (10/21, PR #2 리뷰)**
- [x] scope 정책은 설정된 저장소로만 고른다. 도구 인자 `repo`는 위치만 (§3-8)
- [x] 계속 실패하는 이벤트: `Runtime(max_failures=N)`, `Runtime.skipped`. 하니스는 `None`, 서버는 `CORTEX_MAX_FAILURES`(기본 3) (§3-1)
- [x] 저장소 정책보다 넓은 outcome은 `text`를 비우고 `Event.text_redacted` (§2, §3-8)
- [x] append 위치는 커밋 순서, 판정 기준선 비교는 인메모리 저장소 기준, 한국어 검색은 `simple` (§4, §5)

**합의됨 (10/25)**
- [x] 레포 이름은 `dure-ai/core` (계약·런타임·저장소와 MCP 서버를 함께 담음). 패키지 이름 `cortex-mem`·`cortex-mem-server`는 그대로

**임베딩 (v0.2)**
- [x] 임베딩 모델과 차원, 묶음 문턱값은 서버 설정이다. 엔진이 고르지 않는다.
- [ ] 저장 계층의 차원 문제(pgvector 인덱스 한도 등)는 설계 v2 §3.4 참고.

**저장 계층 (PostgreSQL + pgvector)**
- [ ] 임베딩 모델 = 벡터 차원. Upstage 임베딩으로 바꾸면 pgvector 인덱스 한도에 걸릴 수 있다(차원 수치는 확인 필요).
- [ ] 공용 DB 위치: team scope는 공용 DB가 있어야 사람 간 독립 출처를 셀 수 있다.
- **운영 요구**: PostgreSQL 14 이상, pgvector: 정확 검색(기본)은 0.7 이상, `CORTEX_VECTOR_INDEX=hnsw`는 0.8 이상이 필요하다(HNSW 검색은 `SET LOCAL hnsw.iterative_scan = relaxed_order`를 항상 실행한다). 코드는 PG와 pgvector 버전을 검사하지 않으므로 이 버전 요구는 "가정(검사 안 함)"이다. 검사하는 것은 `vector` 타입이 있는지(`require_vector_type`)뿐이다. DB마다 `CREATE EXTENSION vector`가 필요하고, 확장이 없을 때만 시작할 때 빠진 것을 알리고 멈춘다. 드라이버는 선택 의존성 `cortex-mem[postgres]`(psycopg 3).
- [ ] 전역 잠금(DB 하나에 잠금 하나)을 유지할지는 지연을 잰 뒤 정한다. 근사 색인 기본값도 측정 뒤 정한다.

**모두**
- [ ] 독립 출처의 기본 단위를 `user`로 볼지, `(user, session)`으로 볼지. 계약은 둘 다 담을 수 있고, 판정은 엔진 몫이다.

## 8. 기존 연구를 엔진으로 감싸기 (예시)

`add`/`search` API를 가진 외부 메모리 시스템은 자기 저장소를 쓰는 **예외**로 허용한다. 대신 handoff의 `sources`에 원래 이벤트 id를 유지해야 한다. 근거를 잃은 항목은 런타임 필터(§3-5)가 뺀다.

```python
from collections.abc import Mapping

from cortex_mem.contract import Caller, EngineContext, Event, Handoff, Scope


class ExternalMemoryAdapter:
    def __init__(self, config: Mapping[str, str]):
        self.client = make_client(config)  # 외부 시스템 클라이언트

    def observe(self, event: Event, ctx: EngineContext):
        if event.kind == "record":
            self.client.add(event.text, user_id=event.user, metadata={"event_id": event.id})
        return []  # 위키 승격은 하지 않는다

    def tick(self, now, ctx):
        return []

    def handoff(self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6):
        hits = self.client.search(query, user_id=caller.user, limit=k)
        return [
            Handoff(id=h["id"], content=h["text"], sources=(h["metadata"]["event_id"],))
            for h in hits
        ]
```

- `client.add`·`client.search`의 이름과 반환 모양은 시스템마다 다르다. 위 코드는 모양만 보여 준다.
- `user_id=caller.user`로는 personal scope만 흉내 낸다. team scope를 채점하려면 팀 단위 id를 따로 둬야 한다.
- 외부 시스템이 기억을 합치면 `metadata`가 사라져 항목이 필터에 걸릴 수 있다. 하니스는 이를 별도 수치로 보고한다.

## 9. 디스코드 공유용 요약

```
[T5] 엔진 포트 계약 v0.2 (리뷰 반영) — docs/contract.md
- 엔진이 구현할 것: observe(event, ctx) / tick(now, ctx) → 결정 목록, handoff(query, caller, scope, ctx, k)
- 승격·강등은 Promote/Demote로 반환 → 런타임이 근거·범위 검증 후 즉시 지식 베이스에 반영 (갱신=같은 id Promote, Skill=kind="skill", 보관=Demote why="stale")
- 엔진 데이터는 ctx.state(전용 영속 저장소, JSON, 값당 64 KiB)에. 대상 하나당 키 하나 권장. state_version 바꾸면 처음부터 재전달
- 이벤트는 런타임이 커서로 정확히 한 번 전달. 시작 시 전체 재생 없음
- 누출 검사는 런타임 두 곳: 결정 검증 + handoff 필터
- scope: personal < team < org. personal 기록은 팀 지식의 근거가 될 수 없음
- Event에 team 추가, conditions는 읽기 전용, Caller(user, team, session, at, agent, repo, context) 도입
- outcome·human_confirm은 refs(판정 대상 id) 없으면 append 거부
- scope 정책은 설정된 저장소(CORTEX_REPO)로만. 정책보다 넓게 공개되는 outcome은 text를 비우고 text_redacted
- 계속 실패하는 이벤트는 Runtime(max_failures=N)이 건너뛰고 skipped에 남김(하니스는 None = 건너뛰지 않음)
- 하니스는 같은 Runtime + InMemoryDatabase로 실행, handoff로 채점. 결정 기록(verdicts)은 관찰 지표
- v0.2: 코어가 임베딩을 색인하고 (scope, owner)별로 묶는다. 엔진은 읽기 전용 ctx.similar(search·neighbors·knowledge·group)를 쓴다. 임베딩 미설정이면 None이라 엔진이 처리해야 한다. 이벤트는 색인된 뒤 전달되고, 색인 버전이 바뀌면 처음부터 재전달
- 저장소 공통 테스트(cortex_mem.testing): PG 구현도 이걸 상속해 통과하면 계약 준수
```
