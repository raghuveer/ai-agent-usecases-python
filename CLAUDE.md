# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status: built out — 10 use cases × 4 approaches (40 projects)

All four approach folders are populated: `raw-api/`, `langchain/`, `langgraph/`, and `claude-agent-sdk/`, each with 10 use-case projects plus a `_template/`.

- `agent-development-guide.md` — the authoritative reference: 10 agent use cases, the implementation approaches, scalability and air-gap considerations, and a per-use-case approach-suitability matrix. (Predates the 4th approach; `SPEC.md` §7 and `TRACKING.md` carry the `claude-agent-sdk` deltas.)
- `ai-usecases-notes.txt` — the user's intent and working agreement for how the repo should be built out.

**Verify, don't assume.** Never claim tests pass without running them. Real commands are per-project and documented in each folder's README (`python -m uv venv`, then `.venv/Scripts/python.exe -m pytest tests/test_unit.py -q`).

## The plan (from `ai-usecases-notes.txt`)

Implement the **10 use cases** from the guide in **each of the three approaches**, using **Python + FastAPI**. The end goal is a public GitHub repo others can clone and run, so each use case must be self-contained and runnable in isolation.

### Required directory layout

```
raw-api/           <usecase>/   # full self-contained project per use case
langchain/         <usecase>/
langgraph/         <usecase>/
claude-agent-sdk/  <usecase>/
```

Each approach folder gets one subfolder per use case (10 total per approach, 40 projects total). The *complete* project structure for a use case lives inside its own subfolder — do not share code across use-case folders, since each must be independently clonable/runnable. This deliberately duplicates `settings.py` / `agent.py` across projects; that is the convention, not an oversight. Include unit tests, and integration tests where applicable, with each project.

### The 10 use cases (see guide §1 for full specs)

1. Q&A / RAG Chatbot
2. Code Generation Agent
3. Data Extraction (structured output)
4. Research Agent
5. Customer Support Triage Agent
6. SQL / DB Agent
7. Multi-Agent Orchestration
8. Autonomous Workflow (ReAct / Plan-Act-Reflect)
9. Personalised Recommendations
10. Human-in-the-Loop (HITL) Approval Workflow

## Approach selection — this is the core architectural decision

The point of the repo is to **contrast** the four approaches, so each use case is implemented in all four even where one is a poor fit. Use the suitability matrix (guide §2, plus `TRACKING.md`) to set expectations and shape the implementation:

- **`raw-api/`** — Call the LLM directly via an OpenAI-compatible `/v1/chat/completions` endpoint. You build the agent loop, memory, and state yourself. Best-fit: 1, 2, 3, 6, 9. For 7, 8, 10 the value is showing the DIY plumbing that the frameworks abstract away.
- **`langchain/`** — Chains, document loaders, retrievers, memory backends, tool wrappers. Best-fit: 1–6, 9 (rapid prototyping, medium complexity).
- **`langgraph/`** — Typed state graph with cycles, conditional edges, parallel branches, and native `interrupt()` for HITL. Best-fit: 7, 8, 10 (multi-agent, ReAct loops, **durable** approval gates).
- **`claude-agent-sdk/`** — The Python `claude-agent-sdk` package: the SDK supplies the agent loop, built-in tools (Read/Write/Edit/Bash/Glob/Grep/WebSearch/WebFetch), subagents, hooks, and a permission callback. Best-fit: 2, 7, 8, 10. Poor fit for one-shot jobs (3, 9), where no loop runs.

When implementing, make the differences legible: the raw-api version should expose exactly what is sent to the LLM; the langgraph version should lean on graph cycles / `interrupt()` rather than reimplementing them; the claude-agent-sdk version should lean on built-in tools, `agents={}`, and `can_use_tool` rather than hand-writing loops or orchestration.

### claude-agent-sdk specifics (easy to get wrong)

- **It is the Python SDK**, and all app code is Python — but the SDK spawns the **Claude Code CLI (Node)** as a subprocess. Node 18+ and the CLI must be on PATH for live runs; unit tests need neither.
- **No base-URL/key arguments.** Config reaches the SDK through the subprocess env: `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`. Use `ANTHROPIC_AUTH_TOKEN` — `ANTHROPIC_API_KEY` makes the CLI send `x-api-key`, which the gateway rejects with 401.
- **`LLM_BASE_URL` has no `/v1` suffix here** (Anthropic surface; the SDK appends `/v1/messages`). The other three approaches do include it.
- **`setting_sources=[]` is mandatory** so runs never inherit `~/.claude` or the repo's `.claude/`.
- **Mock at the `runner` seam**, never by spawning the CLI. `collect()` is pure over public SDK types and is tested against real `AssistantMessage`/`ResultMessage` objects.
- **No free-local fallback** — see the cost rules below.

## Conventions to honor

- **Language/framework:** Python + FastAPI for every project.
- **LLM access:** assume an OpenAI-compatible proxy endpoint (LiteLLM), not a hardcoded provider — this keeps examples portable and air-gap-friendly (guide §5). Use the verified local platform below.
- **Public-repo readiness:** each use-case folder should be independently understandable and runnable — its own README/run instructions, dependency manifest, and tests. No secrets in code; configure via env vars.

## Verified local platform access (re-checked 2026-07-29)

An **AI Utility Platform** (`aiup-*` microservices) runs locally under Rancher Desktop. All examples target it via env-configured base URL + Bearer key — never hardcode keys or providers.

| Need | Endpoint | Surface |
|---|---|---|
| **Gateway front door** (guardrails + usage tracking) | `http://localhost:8094` | OpenAI: `/v1/chat/completions` · Anthropic: `/v1/messages` |
| Ollama (direct, self-hosted) | `http://localhost:11434` | native `/api/*` + OpenAI `/v1/*` |

> ⚠️ **Changed since 2026-06-24.** The gateway moved **`:8080` → `:8094`**, and the **LiteLLM tier is gone** (ADR 0036): `projects-ms` now mints native `sk-aiup-…` virtual keys itself, and `:4000` no longer exists. **Model aliases dropped their version suffixes** — `claude-haiku` / `claude-sonnet`, not `claude-haiku-4-5` / `claude-sonnet-4-6`. Old virtual keys are rejected (`invalid_virtual_key`). The 30 pre-`claude-agent-sdk` projects still ship the old values in `.env.example`.

- **Auth:** `Authorization: Bearer <key>` — the gateway **rejects `x-api-key`**. Clients use a platform virtual key (`sk-aiup-…`), read from env `LLM_GATEWAY_KEY`; do not commit it. Mint one with `scripts/seed-virtual-keys.mjs` in the platform repo (`E:\minfy-github\ai-utility-infrastructure\ai-utility-platform`).
- **Self-hosted models (zero marginal cost — the default for dev + tests where the approach allows):** `qwen3:1.7b` (general), `qwen2.5-coder:1.5b` (code), `qwen3:0.6b` (tiny/fast); heavier `qwen2.5:7b-instruct` / `qwen2.5-coder:7b-instruct` available. Gateway aliases: `qwen-local-instruct` / `qwen-local-coder`. **Disable qwen3 "thinking" mode** (`"think": false` on Ollama, or `/no_think`) to save tokens/latency.
- **Anthropic models (limited budget — use sparingly):** `claude-haiku` is the default cloud model; `claude-sonnet` opt-in for hard cases only; **do not use Opus** (budget).

### Cost discipline (the budget is small and real)
- Default every app and **all automated tests to local Qwen** — *except* `claude-agent-sdk/`, which cannot: the SDK drives the Claude Code harness and small local models cannot sustain it (including `qwen3:0.6b`). That approach defaults to `claude-haiku`, escalating to `claude-sonnet` only where documented.
- **Every `claude-agent-sdk` run must cap both `max_turns` and `max_budget_usd`.** An agent loop makes an unbounded number of model calls; a token cap alone is not enough.
- **Unit tests must not hit the network** — mock the LLM client (or, for `claude-agent-sdk`, inject the `runner` stub; never spawn the CLI). Integration tests default to local Qwen where possible; gate cloud tests behind `RUN_ANTHROPIC_TESTS=1`. All `claude-agent-sdk` integration tests are double-gated (`RUN_INTEGRATION=1` + `RUN_ANTHROPIC_TESTS=1`).
- See `SPEC.md` (full contract, incl. §7 agent-SDK deltas), `PLAN.md` (build order), and `TRACKING.md` (per-usecase × approach feasibility — including combos flagged impractical or overkill).

## Workflow vs agent distinction (guide §4)

When implementing, be explicit about whether a use case is a deterministic **workflow** (hardcoded control flow) or an **agent** (LLM decides next action). In LangGraph this maps directly to graph shape: workflows have no cycles and only unconditional edges; agents have cycles and conditional edges. Most real use cases here are hybrid (deterministic outer flow, agentic inner loop) — model them that way.
