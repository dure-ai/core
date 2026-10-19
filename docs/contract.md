# 엔진 포트 계약 v0.1.0

> 상태: 초안 (팀 검토 전). 코드 정본은 `src/cortex_mem/contract.py`.
> 이 문서의 규칙 중 타입으로 표현되지 않는 것(호출 순서, id·scope 의미)도 계약의 일부다.

## 1. 무엇이 고정이고 무엇이 교체되나

```
에이전트 ─MCP→ 서버 ──append──→ EpisodeStore (고정, append-only)
                 │                    ▲ 읽기 전용(EpisodeReader)
                 └──observe/handoff──→ Engine (교체 가능: A·B·C·기준선)
하니스 ─(같은 Event 흐름)──→ 위와 동일한 경로
```

- 각자 만드는 건 `Engine` 하나. 서버와 하니스는 엔진을 `EngineFactory = Callable[[EpisodeReader], Engine]`로만 만든다. 즉 생성자는 `__init__(self, reader)` 하나.
- 엔진은 저장소를 **읽기만** 한다(`get`, `search`). 쓰기는 서버·하니스만 한다.

## 2. 타입

### Event — 관찰 하나

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str | 호출자가 정한다(서버: uuid, 하니스: 시드에서 결정적으로). 중복이면 `append`가 `ValueError` |
| `kind` | `record` · `recall` · `outcome` · `human_confirm` · `commit` | |
| `at` | datetime | **timezone 필수**. 없으면 생성 시 `ValueError` |
| `user`, `session` | str | 매 호출마다 받는다. 독립 출처를 세는 기본 단위 |
| `scope` | `personal` · `team` | 이 기록을 누구에게 보일지 (§3-3) |
| `text` | str | record: 기록 내용, recall: 질의, commit: 커밋 메시지 |
| `agent` | str? | 클라이언트 종류 (claude-code, cursor, codex …) |
| `signal` | `self` · `test` · `review` · `human`? | 증거 신호 종류 |
| `outcome` | `success` · `failure` · `unknown`? | outcome 이벤트의 결과 |
| `conditions` | Mapping[str, str] | **이 주장이 참인 조건만**. 예: `{"service": "A", "db": "cloud_sql"}` |
| `refs` | tuple[str, ...] | 다른 항목을 가리키는 id (§3-2) |
| `repo`, `rev` | str? | 근거가 나온 곳(저장소, 커밋·CI 실행 id). `conditions`와 섞지 않는다 |

### QueryContext

`user`, `session`, `scope`, `conditions` — 지금 질의하는 쪽의 맥락.

### Handoff — 엔진이 에이전트 컨텍스트에 넣는 항목 하나

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | str | 엔진이 정해도 된다. 단 **호출 사이에 안정적**이어야 한다(recall 이벤트의 `refs`로 돌아옴) |
| `content` | str | 에이전트에게 보일 내용 |
| `sources` | tuple[str, ...] | 근거 이벤트 id. 모두 `reader.get`으로 찾을 수 있어야 한다 |
| `status` | `memory` · `knowledge` · `invalidated`? | **선택**. 에이전트 표시용이며 채점에는 쓰지 않는다 (하니스 원칙 1) |
| `conditions` | Mapping[str, str] | 이 항목이 유효한 조건 |
| `confidence` | float? | 0..1. 주면 Brier 점수로 확률 채점 |

### 프로토콜

```python
class EpisodeReader(Protocol):          # 엔진이 받는 것
    def get(self, ids) -> list[Event]: ...
    def search(self, query, ctx, k=20, kinds=None) -> list[tuple[Event, float]]: ...

class EpisodeStore(EpisodeReader, Protocol):   # 서버·하니스가 쓰는 것 (저장 계층 담당 구현 대상)
    def append(self, event) -> str: ...

class Engine(Protocol):                  # 필수
    def observe(self, event) -> None: ...
    def handoff(self, query, ctx, k=6) -> list[Handoff]: ...

class KnowledgeSnapshot(Protocol):       # 선택 어댑터
    def snapshot(self, at) -> list[Handoff]: ...
```

`Engine`과 `KnowledgeSnapshot`은 `runtime_checkable`이라 하니스가 `isinstance`로 구현 여부를 확인한다.

## 3. 규칙

### 3-1. 호출 순서

1. 모든 이벤트는 `store.append(event)` → `engine.observe(event)` 순서다. 그래서 엔진은 `observe` 안에서 그 이벤트를 `get`할 수 있다.
2. recall 한 번은 이렇게 흐른다.
   ```
   handoffs = engine.handoff(query, ctx, k)
   event = Event(kind="recall", text=query, refs=tuple(h.id for h in handoffs), ...)
   store.append(event); engine.observe(event)
   return handoffs  # 에이전트에게
   ```
   엔진은 자기가 건넨 것이 무엇인지 recall 이벤트로 다시 관찰한다(엔진 B 계보 추적용).

### 3-2. `refs` 의미

| kind | refs |
|---|---|
| `recall` | 에이전트에게 건넨 Handoff id |
| `outcome`, `human_confirm` | 판정 대상인 이벤트 또는 Handoff id |
| `record` | 이 기록을 쓰기 직전에 건네받은 Handoff id (알면). 에이전트가 받은 걸 다시 말한 건지 구분하는 단서 |
| `commit` | 비워 둔다 |

### 3-3. scope 가시성 (`search`)

- `ctx.scope == "personal"` → `ctx.user`의 이벤트만.
- `ctx.scope == "team"` → `ctx.user`의 이벤트 + 모든 사용자의 `team` 이벤트. 다른 사람의 `personal` 이벤트는 절대 보이지 않는다.

### 3-4. 결정성

같은 이벤트 순서를 넣으면 `handoff`, `snapshot`은 같은 결과를 내야 한다. 하니스가 재실행 뒤집힘률(목표 10% 이하)을 잰다. 엔진 안에서 난수를 쓰면 시드를 고정한다.

### 3-5. `snapshot(at)`

`at`까지의 이벤트를 모두 관찰한 직후에 호출된다. "지금 이 순간 에이전트에게 건넬 수 있는 것 전체"를 돌려준다. 시간에 따라 만료되는 엔진은 `at`을 평가 시점으로 쓴다.

## 4. 하니스 이벤트 ↔ Event 대응

하니스 초안 §2-3의 이벤트 종류를 이렇게 담는다.

| 하니스 이벤트 | kind | 주요 필드 |
|---|---|---|
| 에이전트 기록 | `record` | `text`, `agent`, `signal="self"`, `conditions` |
| recall 요청과 응답 내용 | `recall` | `text`=질의, `refs`=건넨 Handoff id |
| CI·테스트 결과 | `outcome` | `signal="test"`, `outcome`, `refs`, `repo`, `rev` |
| 사람 확인 | `human_confirm` | `signal="human"`, `outcome`, `refs` |
| 대상 코드 커밋(여러 저장소) | `commit` | `repo`, `rev`, `text`=커밋 메시지 |

## 5. v0.2 이후로 미룬 것

- **LLM 주입**: 엔진이 LLM을 쓰면 하니스가 모델 버전과 호출 예산을 정해 주입한다(하니스 초안). 모델·예산이 정해지면 `EngineFactory`에 인자를 추가한다.
  - 제공자는 Upstage, 예산은 팀 크레딧 합계 $330(1인 $110). 엔진·실행별 예산 배분은 하니스 설계에서 정한다.
  - **기존 연구 채점이 시작되기 전까지 필요하다.** 기존 메모리 시스템 대부분이 저장할 때 LLM을 부르므로, 주입 없이는 "모든 엔진에 같은 모델·같은 예산" 조건을 지킬 수 없다.
- 계약을 바꿀 때는 `CONTRACT_VERSION`을 올리고 근거를 이 문서에 남긴다(하니스 초안 "계약 확장 절차").

## 6. 팀 확인 필요

담당은 역할 기준으로 적는다. 사람별 담당은 팀 회의에서 확정한다.

**하니스 설계**
- [ ] §4 대응표가 세계 생성기 출력과 맞는지. 특히 "관찰자 간 따라하기 관계"를 `record.refs`로 표현해도 되는지.
- [ ] §3-4 결정성 규칙과 §3-5 `snapshot(at)` 호출 시점이 채점 방식과 맞는지.
- [ ] LLM 예산($330)을 엔진·실행별로 어떻게 나눌지.

**기존 연구 채점**
- [ ] 채점할 시스템을 §7 어댑터 모양으로 감쌀 수 있는지.
- [ ] 외부 시스템이 기억을 합치거나 바꿔 쓰면 근거 이벤트 id를 잃는다. 이때 `sources`를 빈 tuple로 허용할지.

**저장 계층 (PostgreSQL + pgvector)**
- [ ] `tests/test_store.py`의 `STORES`에 구현을 추가해 통과시키면 계약을 지킨 것이다. `search` 점수는 "높을수록 좋음"만 약속한다.
- [ ] 임베딩 모델 = 벡터 차원. 현재 `potion-base-8M`(256차원, 영어 기준). Upstage 임베딩으로 바꾸면 차원이 커져 pgvector 인덱스 한도에 걸릴 수 있다(차원 수치는 확인 필요).
- [ ] 한국어 전문 검색. PostgreSQL 기본 설정에는 한국어가 없다. `simple` 설정으로 갈지, pg_bigm 같은 확장을 쓸지.
- [ ] `team` scope용 공용 DB를 어디에 둘지. 각자 로컬 DB로는 사람 간 독립 출처를 셀 수 없다.

**모두**
- [ ] `status`를 선택 필드로 둔 것(채점에 안 씀), `snapshot`을 선택으로 둔 것에 동의하는지.
- [ ] 독립 출처의 기본 단위를 `user`로 볼지, `(user, session)`으로 볼지. 계약은 둘 다 담을 수 있고, 판정은 엔진 몫이다.

## 7. 기존 연구를 엔진으로 감싸기 (예시)

`add`/`search` API를 가진 외부 메모리 시스템은 이렇게 감싼다. 외부 시스템은 자기 저장소를 쓰므로 `reader`는 받기만 한다.

```python
from functools import partial

from cortex_mem.contract import Event, Handoff, QueryContext


class ExternalMemoryAdapter:
    def __init__(self, reader, client):
        self.reader = reader  # 외부 시스템은 쓰지 않음
        self.client = client

    def observe(self, event: Event) -> None:
        if event.kind == "record":
            self.client.add(event.text, user_id=event.user, metadata={"event_id": event.id})

    def handoff(self, query: str, ctx: QueryContext, k: int = 6) -> list[Handoff]:
        hits = self.client.search(query, user_id=ctx.user, limit=k)
        return [
            Handoff(id=h["id"], content=h["text"], sources=(h["metadata"]["event_id"],))
            for h in hits
        ]


factory = partial(ExternalMemoryAdapter, client=make_client())  # EngineFactory 모양
```

- `client.add`·`client.search`의 이름과 반환 모양은 시스템마다 다르다. 위 코드는 모양만 보여 준다.
- `user_id=ctx.user`로는 `personal` scope만 흉내 낸다. `team` scope를 채점하려면 팀 단위 id를 따로 둬야 한다.
- 외부 시스템이 기억을 합치면 `metadata`가 사라질 수 있다(§6 "기존 연구 채점" 두 번째 항목).

## 8. 디스코드 공유용 요약

```
[T5] 엔진 포트 계약 v0.1 초안 올렸습니다 (cortex-mem/mcp-server docs/contract.md)
- 각자 만들 건 Engine 하나: observe(event), handoff(query, ctx, k) → list[Handoff]
- 생성자는 __init__(self, reader) 하나. reader는 읽기 전용 저장소(get, search)
- 순서 보장: append → observe. recall 응답도 kind="recall", refs=건넨 id로 다시 observe됨
- Event에 repo/rev 추가(CI·커밋 출처), conditions는 "참인 조건"만
- Handoff.status는 선택 필드, 채점엔 안 씀 / snapshot(at)은 선택 어댑터(Brier 채점용)
- scope: personal=본인 것만, team=본인+전원의 team 기록
- 기준선 NoMemory·PromoteAll 엔진 포함, 테스트로 계약 확인
- 저장소 공통 테스트(tests/test_store.py): PG 구현도 이걸 통과하면 계약 준수
- LLM 주입은 v0.2(Upstage, 예산 $330). 기존 연구 채점 전까지 필요
- 기존 연구를 Engine으로 감싸는 예시는 §7
- 확인 부탁: §6 (하니스 설계 / 기존 연구 채점 / 저장 계층 / 모두)
```
