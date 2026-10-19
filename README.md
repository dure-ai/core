# cortex-mem MCP server

에이전트 기억을 Memory(소문)로 받고, 근거가 쌓이면 Knowledge(사실)로 승격하는 엔진의 MCP 서버와 엔진 포트 계약.

- 엔진 포트 계약: [docs/contract.md](docs/contract.md)
- 라이선스: Apache 2.0. 일부 코드는 같은 저자의 [mem-mcp](https://github.com/Jh-jaehyuk/mem-mcp)(MIT)에서 가져와 Apache 2.0으로 재라이선스했다. [NOTICE](NOTICE) 참고.

## 개발

```bash
uv sync
uv run pytest
```
