# AI Agent Use Cases — Raw API · LangChain · LangGraph

[![unit-tests](https://github.com/raghuveer/ai-agent-usecases-python/actions/workflows/ci.yml/badge.svg)](https://github.com/raghuveer/ai-agent-usecases-python/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/raghuveer/ai-agent-usecases-python?label=release)](https://github.com/raghuveer/ai-agent-usecases-python/releases)
[![security review](https://img.shields.io/badge/security-reviewed-brightgreen)](docs/security-review.md)
[![dependencies: pip-audit](https://img.shields.io/badge/deps-pip--audit%20%2B%20Dependabot-blue)](.github/dependabot.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue)](#)

> Ten common LLM agent use cases — RAG, code-gen, data extraction, research, support triage, SQL, multi-agent, ReAct, recommendations, human-in-the-loop — each implemented three ways: **raw API**, **LangChain**, and **LangGraph**. Python + FastAPI, with unit & integration tests. A practical side-by-side of the three approaches.

Ten common agent use cases, each implemented **three ways** — direct **Raw API**, **LangChain**, and **LangGraph** — in **Python + FastAPI**. The point is to *compare* the approaches on the same problems. Every folder is a self-contained, runnable project with its own tests.

**Other languages:** a TypeScript edition (LangChain.js + LangGraph.js) is planned at `ai-agent-usecases-typescript` _(coming soon)_.

> **Status — v0.2.0:** all 10 use cases built across all 3 approaches (30 projects + a `_template/` per approach), each with offline, mocked **unit tests** and gated **integration tests**. **Security-reviewed** against NIST · OWASP LLM Top 10 · OWASP Web Top 10 (see [`docs/security-review.md`](docs/security-review.md) / [`SECURITY.md`](SECURITY.md)); dependencies patched to current majors, with a CI `pip-audit` gate and Dependabot keeping them current. Examples optimise for clarity of the approach, not production hardening.

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

## Use a different model or provider

Nothing is hardcoded to a provider — every project speaks **OpenAI-compatible HTTP** and reads its model/endpoint from env. To switch, edit `.env` only — **no code changes**:

| To use… | `LLM_BASE_URL` | `LLM_MODEL` | Notes |
|---|---|---|---|
| Bundled gateway (default) | `http://localhost:8080/v1` | `qwen-local-instruct` · `claude-haiku-4-5` | virtual key in `LLM_GATEWAY_KEY` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | `LLM_GATEWAY_KEY` = your OpenAI key |
| Ollama (direct, local) | `http://localhost:11434/v1` | `qwen2.5:7b-instruct` | any non-empty key works |
| Together · Groq · OpenRouter · Azure OpenAI | their `/v1` URL | their model id | all OpenAI-compatible |
| Anthropic · Bedrock · Vertex | a LiteLLM proxy `/v1` | the proxy's alias | LiteLLM normalises these onto the OpenAI surface |

**Tuning knobs** (also env, also no code change): `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.

Two caveats that are about the *model*, not the code:
- **Capability ≠ code:** small local models can't reliably do strict-JSON extraction or multi-step ReAct — that's why use cases 3/4/7/8 default to a stronger model. Point them at a capable model and they work as-is; point them at a weak one and the code is unchanged but quality drops.
- **Model quirks live in one place** — `model_profile()` in each project's `app/llm.py` (e.g. disabling qwen3's "thinking" mode). Add a new model family there in a single spot.

> **Structured-output modes (UC3 prototype):** `03-data-extraction` also honours `LLM_STRUCTURED_MODE=text|native` — `text` is the portable prompt-and-parse path; `native` uses the provider's JSON / structured-output feature for higher reliability where available. See that project's README.

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
- `SECURITY.md` — disclosure policy & scope.
- `docs/security-review.md` — security review vs NIST · OWASP LLM Top 10 · OWASP Web Top 10 (SAST, dependency-CVE, IaC, DAST).

## License & credits

**MIT** — see [`LICENSE`](LICENSE). Copyright (c) 2026 Raghuveer Dendukuri.

**Author:** Raghuveer Dendukuri · **Co-author:** Claude Code (Opus). Every source
file carries an `SPDX-License-Identifier: MIT` header that also names its use case
and links the relevant folder README.
