# AI Agent Use Cases — Raw API · LangChain · LangGraph

Ten common agent use cases, each implemented **three ways** — direct **Raw API**, **LangChain**, and **LangGraph** — in **Python + FastAPI**. The point is to *compare* the approaches on the same problems. Every folder is a self-contained, runnable project with its own tests.

<!-- After pushing, replace <OWNER> to activate the badge:
[![unit-tests](https://github.com/<OWNER>/ai-usecases/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/ai-usecases/actions/workflows/ci.yml) -->

> **Status:** all 10 use cases built across all 3 approaches (30 projects + a `_template/` per approach). Each has offline, mocked **unit tests** and gated **integration tests**. Examples optimise for clarity of the approach, not production hardening.

## Use-case matrix

| # | Use case | raw-api | langchain | langgraph | Runtime model |
|---|----------|:-------:|:---------:|:---------:|---------------|
| 1 | Q&A / RAG chatbot | [▸](raw-api/01-rag) | [▸](langchain/01-rag) | [▸](langgraph/01-rag) | local Qwen |
| 2 | Code generation | [▸](raw-api/02-code-generation) | [▸](langchain/02-code-generation) | [▸](langgraph/02-code-generation) | local Qwen (coder) |
| 3 | Data extraction | [▸](raw-api/03-data-extraction) | [▸](langchain/03-data-extraction) | [▸](langgraph/03-data-extraction) | Haiku¹ |
| 4 | Research agent | [▸](raw-api/04-research-agent) | [▸](langchain/04-research-agent) | [▸](langgraph/04-research-agent) | Haiku¹ |
| 5 | Customer support triage | [▸](raw-api/05-support-triage) | [▸](langchain/05-support-triage) | [▸](langgraph/05-support-triage) | local Qwen |
| 6 | SQL / DB agent | [▸](raw-api/06-sql-agent) | [▸](langchain/06-sql-agent) | [▸](langgraph/06-sql-agent) | local Qwen (coder) |
| 7 | Multi-agent orchestration | [▸](raw-api/07-multi-agent) | [▸](langchain/07-multi-agent) | [▸](langgraph/07-multi-agent) | Haiku¹ |
| 8 | Autonomous ReAct | [▸](raw-api/08-autonomous-react) | [▸](langchain/08-autonomous-react) | [▸](langgraph/08-autonomous-react) | Haiku¹ |
| 9 | Recommendations | [▸](raw-api/09-recommendations) | [▸](langchain/09-recommendations) | [▸](langgraph/09-recommendations) | local Qwen |
| 10 | Human-in-the-loop approval | [▸](raw-api/10-hitl-approval) | [▸](langchain/10-hitl-approval) | [▸](langgraph/10-hitl-approval) | local Qwen |

¹ **Why Haiku for 3, 4, 7, 8?** Building these we found small local models can't reliably emit strict schema JSON (extraction) or drive a multi-step text **ReAct** loop (research / ReAct / multi-agent) — they wandered into prose and skipped tools. Those four default to `claude-haiku-4-5`; the rest run on free self-hosted Qwen. See `TRACKING.md` for the full feasibility matrix and `SPEC.md` for the cost rules.

Which approach suits which use case (and the two raw-api combos deliberately kept minimal — multi-agent and HITL — with "why impractical here" notes) is tracked in **`TRACKING.md`**.

## How models are accessed

Every example speaks **OpenAI-compatible HTTP** to a gateway — no provider is hardcoded. Configure via env; each project ships a `.env.example`:

| Var | Example | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway base URL |
| `LLM_GATEWAY_KEY` | `sk-…` | `Authorization: Bearer` virtual key |
| `LLM_MODEL` | `qwen-local-instruct` / `qwen-local-coder` / `claude-haiku-4-5` | allow-listed model alias |

Copy `.env.example` → `.env` (gitignored) and fill in your key. The `.env` is **never committed**; only `.env.example` is.

## Quick start (any project)

Each project is independent and uses [`uv`](https://docs.astral.sh/uv/):

```bash
cd raw-api/01-rag                 # pick any of the 30
cp .env.example .env              # then edit .env: set LLM_GATEWAY_KEY (and LLM_MODEL if needed)

uv venv
# On Windows, pin uv to the venv interpreter explicitly:
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
#   (Linux/macOS: just `uv pip install -e ".[dev]"`)

# Offline unit tests — mocked LLM, no network, no key needed:
uv run pytest -q -m "not integration"

# Live integration test (needs the gateway + a real key in .env):
RUN_INTEGRATION=1 uv run pytest -q -m integration

# Serve the API:
uv run uvicorn app.main:app --reload
# → GET /health, POST /run   (see the project README for the request shape)
```

## Testing model

- **Unit tests** — the LLM is mocked; fully offline; these run in CI.
- **Integration tests** — call the live gateway; gated behind `RUN_INTEGRATION=1`. The four Haiku use cases also mark theirs `anthropic` since they spend a small, capped budget.

CI (`.github/workflows/ci.yml`) runs the **unit tests only** across all projects on every push/PR — no network to any model, no keys required.

## Docs

- `agent-development-guide.md` — the 10 use cases, the three approaches, scalability, language support.
- `SPEC.md` — project contract, model strategy, testing & cost rules.
- `PLAN.md` — phased build order and final status.
- `TRACKING.md` — feasibility matrix & per-cell build status.

## License & credits

**MIT** — see [`LICENSE`](LICENSE). Copyright (c) 2026 Raghuveer Dendukuri.

**Author:** Raghuveer Dendukuri · **Co-author:** Claude Code (Opus). Every source
file carries an `SPDX-License-Identifier: MIT` header that also names its use case
and links the relevant folder README.
