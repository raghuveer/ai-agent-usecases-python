# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status: greenfield / planning stage

This repository currently contains **only planning artifacts** — there is no application code, build tooling, or tests yet. The three approach folders (`raw-api/`, `langchain/`, `langgraph/`) are intentionally empty placeholders waiting to be populated.

- `agent-development-guide.md` — the authoritative reference: 10 agent use cases, the three implementation approaches, scalability and air-gap considerations, and a per-use-case approach-suitability matrix. **Read this before implementing any use case** — it dictates which approach fits which use case.
- `ai-usecases-notes.txt` — the user's intent and working agreement for how the repo should be built out.

When generating code, do not invent commands or claim tests pass — there is no harness yet. Establish conventions (and document the real commands here) as the first use case is built.

## The plan (from `ai-usecases-notes.txt`)

Implement the **10 use cases** from the guide in **each of the three approaches**, using **Python + FastAPI**. The end goal is a public GitHub repo others can clone and run, so each use case must be self-contained and runnable in isolation.

### Required directory layout

```
raw-api/    <usecase>/   # full self-contained project per use case
langchain/  <usecase>/
langgraph/  <usecase>/
```

Each approach folder gets one subfolder per use case (10 total per approach, 30 projects total). The *complete* project structure for a use case lives inside its own subfolder — do not share code across use-case folders, since each must be independently clonable/runnable. Include unit tests, and integration tests where applicable, with each project.

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

The point of the repo is to **contrast** the three approaches, so each use case is implemented in all three even where one is a poor fit. Use the suitability matrix (guide §2) to set expectations and shape the implementation:

- **`raw-api/`** — Call the LLM directly via an OpenAI-compatible `/v1/chat/completions` endpoint (LiteLLM / AI gateway in front). You build the agent loop, memory, and state yourself. Best-fit use cases: 1, 2, 3, 6, 9. For 7, 8, 10 the value is showing the DIY plumbing (manual tool loop, manual pause/resume queue) that the frameworks abstract away.
- **`langchain/`** — Chains, document loaders, retrievers, memory backends, tool wrappers. Best-fit: 1–6, 9 (rapid prototyping, medium complexity).
- **`langgraph/`** — Typed state graph with cycles, conditional edges, parallel branches, and native `interrupt()` for HITL. Best-fit: 7, 8, 10 (multi-agent, ReAct loops, approval gates).

When implementing, make the differences legible: the raw-api version should expose exactly what is sent to the LLM; the langgraph version should lean on graph cycles / `interrupt()` rather than reimplementing them.

## Conventions to honor

- **Language/framework:** Python + FastAPI for every project.
- **LLM access:** assume an OpenAI-compatible proxy endpoint (LiteLLM), not a hardcoded provider — this keeps examples portable and air-gap-friendly (guide §5). Use the verified local platform below.
- **Public-repo readiness:** each use-case folder should be independently understandable and runnable — its own README/run instructions, dependency manifest, and tests. No secrets in code; configure via env vars.

## Verified local platform access (checked 2026-06-24)

An **AI Utility Platform** (`aiup-*` microservices) runs locally under Rancher Desktop. All examples target it via env-configured base URL + Bearer key — never hardcode keys or providers.

| Need | Endpoint | Surface |
|---|---|---|
| **Recommended front door** (guardrails + usage tracking) | `http://localhost:8080` | OpenAI: `/v1/chat/completions` · Anthropic: `/v1/messages` |
| LiteLLM proxy (direct) | `http://localhost:4000/v1/...` | OpenAI-compatible |
| Ollama (direct, self-hosted) | `http://localhost:11434` | native `/api/*` + OpenAI `/v1/*` |

- **Auth:** `Authorization: Bearer <key>`. The Anthropic API key lives server-side in LiteLLM — clients use a **platform virtual key** (prefer a per-project key; the LiteLLM master key works for local dev). Read from env `LLM_GATEWAY_KEY`; do not commit it.
- **Self-hosted models (zero marginal cost — the default for dev + tests):** `qwen3:1.7b` (general), `qwen2.5-coder:1.5b` (code), `qwen3:0.6b` (tiny/fast); heavier `qwen2.5:7b-instruct` / `qwen2.5-coder:7b-instruct` available. Via LiteLLM use aliases `qwen-local-instruct` / `qwen-local-coder`. **Disable qwen3 "thinking" mode** (`"think": false` on Ollama, or `/no_think`) to save tokens/latency.
- **Anthropic models (limited budget — use sparingly):** `claude-haiku-4-5` is the default cloud model; `claude-sonnet-4-6` opt-in for hard cases only; **do not use Opus** (budget).

### Cost discipline (the budget is small and real)
- Default every app and **all automated tests to local Qwen**. Anthropic calls must be deliberate and capped (`max_tokens`).
- **Unit tests must not hit the network** — mock the LLM client. Integration tests default to local Qwen; gate the few Haiku tests behind an opt-in env flag (e.g. `RUN_ANTHROPIC_TESTS=1`).
- See `SPEC.md` (full contract), `PLAN.md` (build order), and `TRACKING.md` (per-usecase × approach feasibility — including combos flagged impractical on local-only models).

## Workflow vs agent distinction (guide §4)

When implementing, be explicit about whether a use case is a deterministic **workflow** (hardcoded control flow) or an **agent** (LLM decides next action). In LangGraph this maps directly to graph shape: workflows have no cycles and only unconditional edges; agents have cycles and conditional edges. Most real use cases here are hybrid (deterministic outer flow, agentic inner loop) — model them that way.
