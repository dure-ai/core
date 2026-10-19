# cortex-mem MCP server

에이전트 기억을 Memory(소문)로 받고, 근거가 쌓이면 Knowledge(사실)로 승격하는 엔진의 MCP 서버와 엔진 포트 계약.

- 엔진 포트 계약: [docs/contract.md](docs/contract.md)
- 라이선스: Apache 2.0. 일부 코드는 같은 저자의 [mem-mcp](https://github.com/Jh-jaehyuk/mem-mcp)(MIT)에서 가져와 Apache 2.0으로 재라이선스했다. [NOTICE](NOTICE) 참고.

## 개발

```bash
uv sync
uv run pytest
```

## Claude Code에 등록

```bash
claude mcp add cortex-mem -- uv run --directory /path/to/mcp-server python -m cortex_mem.server
```

도구: `record_episode`(기록), `recall`(불러오기), `report_outcome`(불러온 기억이 도움됐는지 보고). 서버 프로세스 하나가 세션 하나다.

| 환경변수 | 기본값 | |
|---|---|---|
| `CORTEX_DB_PATH` | `~/.cortex-mem/episodes.db` | 이벤트 저장소(SQLite) |
| `CORTEX_USER` | OS 사용자 | 기록자 |
| `CORTEX_ENGINE` | `promote_all` | `promote_all` \| `no_memory` |
