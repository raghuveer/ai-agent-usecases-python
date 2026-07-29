# UC04 — research-agent (claude-agent-sdk)

An iterative research agent: search, read, refine, synthesise, cite.

## Two modes, offline by default

The Agent SDK ships `WebSearch` and `WebFetch` as **built-in** tools — no search
API client, no HTML extraction, no citation plumbing. But this repository must
also run air-gapped, so web access is opt-in:

| Mode | `RESEARCH_ALLOW_WEB` | Tools | Network |
|---|---|---|---|
| `offline` (default) | `0` | `Grep`, `Glob`, `Read` | none |
| `web` | `1` | the above **+** `WebSearch`, `WebFetch` | yes |

Defaulting to offline is deliberate: a live-web default would make the example
non-reproducible and would quietly fail on an isolated host. In offline mode the
web tools are **absent from the allow-list entirely** — the agent cannot reach
the network even if it decides it wants to. A unit test asserts exactly that.

The active mode is reported on both `GET /health` and every `/run` response, so
a reader never has to guess which one ran.

| Concern | raw-api / langchain / langgraph | claude-agent-sdk |
|---|---|---|
| Web search | pluggable search-tool client you write | built-in `WebSearch` |
| Fetching a page | HTTP + HTML→text extraction | built-in `WebFetch` |
| Air-gap story | ship a mock/local search tool | drop the web tools from the allow-list |
| Loop | hand-written ReAct or graph cycle | the agent loop, `max_turns`-bounded |

## Citations are observed, not claimed

`citations` is built from what the agent actually opened — filenames from `Read`,
URLs from `WebFetch` — not from prose it wrote. A source it mentions but never
fetched does not appear. `searches` likewise records the `Grep` patterns and
`WebSearch` queries it tried, which makes its strategy inspectable.

## Corpus

`data/` holds three short markdown notes from this repository's own build
findings.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"04-research-agent","mode":"offline"|"web"}`
- `POST /run` body `{"question": str}` →
  `{"answer": str, "mode": str, "citations": [str], "searches": [str],
    "tools_used": [str], "num_turns": int, "cost_usd": float}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | `claude-sonnet` for harder questions |
| `RESEARCH_ALLOW_WEB` | `0` | `1` enables the built-in web tools |
| `AGENT_MAX_TURNS` | `12` | raised to ≥8: research is iterative |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | raise for harder questions |

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent, offline retrieval (gateway + key + Node/CLI):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"question":"What did we learn about local models and tool use?"}'
```
