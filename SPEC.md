# Specification — 10 Agent Use Cases × 4 Approaches

This spec governs the 40 example projects (`<approach>/<usecase>/`). Read alongside `agent-development-guide.md` (use-case definitions), `CLAUDE.md` (platform access + cost rules), `PLAN.md` (build order), and `TRACKING.md` (feasibility matrix).

## 1. Goals & non-goals

**Goals:** a public, clonable reference where each use case is implemented in raw-api, langchain, langgraph, and claude-agent-sdk so readers can *compare* the four approaches; each project independently runnable with its own tests; cost kept near-zero by defaulting to self-hosted Qwen **wherever the approach permits it** (the Agent SDK does not — see §2).

**Non-goals:** production hardening, auth/multi-tenancy, polished UIs, exhaustive provider coverage. Examples optimise for **clarity of the approach**, not feature completeness.

**Dev vs runtime (do not conflate):** *Authoring* this code is done by Claude Code on the maintainer's Claude Max subscription — it never touches the platform's Anthropic budget. The **AI Utility Platform is solely the runtime LLM target the example apps call** to integrate and test approach/usecase behavior. So the budget rules below constrain the *apps' calls*, not the development process.

## 2. Model strategy

| Tier | Model | Use it for |
|---|---|---|
| Default (local, free) | `qwen3:1.7b` | general chat/reasoning, classification, routing |
| Local code | `qwen2.5-coder:1.5b` | code generation, SQL |
| Local tiny | `qwen3:0.6b` | smoke tests, latency demos |
| Local heavy | `qwen2.5:7b-instruct` / `qwen2.5-coder:7b-instruct` | when 1.5–1.7b tool-calling/JSON is too unreliable but you want to stay local |
| Cloud default | `claude-haiku` | the few cases where small local models can't deliver reliable tool-calls / structured output — **and every `claude-agent-sdk` project** |
| Cloud opt-in | `claude-sonnet` | hardest reasoning/multi-agent only, behind an env flag |
| Excluded | Opus | budget — do not use |

Model id is **always** resolved from env (`LLM_MODEL`), with a per-project sensible default. Apps never hardcode a provider — they speak to the gateway via `LLM_BASE_URL`.

> **Platform note (2026-07-29):** the gateway listens on **`:8094`** and the LiteLLM tier has been removed (ADR 0036) — `projects-ms` mints native `sk-aiup-…` virtual keys, and allow-listed aliases dropped their version suffixes (`claude-haiku`, `claude-sonnet`, `qwen-local-instruct`, `qwen-local-coder`). The 30 pre-`claude-agent-sdk` projects still ship the old `:8080` + suffixed aliases in `.env.example`.

### Framework wiring
- **raw-api:** `httpx`/`openai` SDK pointed at `LLM_BASE_URL`. The agent loop, memory, and state are hand-written (this is the point).
- **langchain / langgraph:** use `langchain-openai`'s `ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_GATEWAY_KEY, model=LLM_MODEL)` for **both** Qwen and Claude — one uniform client path keeps the comparison honest and avoids per-provider branching.
- **claude-agent-sdk:** the `claude-agent-sdk` Python package. It does **not** accept a base URL or key as arguments — it spawns the Claude Code CLI, which reads `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` from its environment. Each project's `app/agent.py` translates the repo-standard `LLM_*` settings into those. Two constraints follow:
  - `LLM_BASE_URL` carries **no `/v1` suffix** (Anthropic surface; the SDK appends `/v1/messages`).
  - Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY` — the gateway requires `Authorization: Bearer` and rejects `x-api-key` with 401.
  - `setting_sources=[]` is mandatory, so runs never inherit the developer's `~/.claude` or the repo's `.claude/`.

### Approach-specific cost rule (claude-agent-sdk)
This approach **cannot** default to free local Qwen — the SDK drives the Claude Code harness, which small local models cannot sustain. It therefore defaults to `claude-haiku` for every use case, with `claude-sonnet` as the documented escalation. To keep that safe, every project MUST set both `max_turns` and `max_budget_usd` on every run, and **all** its integration tests are double-gated behind `RUN_INTEGRATION=1` *and* `RUN_ANTHROPIC_TESTS=1`.

## 3. Shared project contract

Every `<approach>/<usecase>/` folder MUST contain:

```
README.md            # what it does, which approach trade-offs it shows, how to run, env vars, feasibility notes
pyproject.toml       # or requirements.txt — pinned deps
.env.example         # LLM_BASE_URL, LLM_GATEWAY_KEY, LLM_MODEL, usecase-specific vars (no real secrets)
app/                 # FastAPI app: main.py (routes) + the agent/chain/graph logic
tests/
  test_unit.py       # mocked LLM — no network, runs in CI
  test_integration.py# hits local Qwen by default; Anthropic tests marked + gated
```

- **API shape:** expose at least `GET /health` and `POST /run` (use-case-specific request/response Pydantic models). HITL (UC10) adds `POST /resume`.
- **Config:** one `settings.py` using `pydantic-settings` reading the env vars above. No secret literals.
- **Determinism for tests:** the LLM client is injected/overridable so unit tests swap in a stub. In `claude-agent-sdk` the injected seam is a **`runner` callable** (`(prompt, options) -> AgentResult`) rather than a client object, because `query()` spawns a subprocess that unit tests must never launch. The message-parsing function `collect()` stays pure over public SDK types so it can still be tested against real `AssistantMessage`/`ResultMessage` objects.
- **Runtime prerequisite (claude-agent-sdk only):** Node.js 18+ and the Claude Code CLI on PATH for live runs. Unit tests must not require either.

## 4. Testing & cost rules

- **Unit:** mock the model client; assert routing/parsing/loop logic. Zero network. These run in CI.
- **Integration:** default `LLM_MODEL` = local Qwen; assert the app round-trips against the live local model. Mark Anthropic-requiring tests with `@pytest.mark.anthropic` and skip unless `RUN_ANTHROPIC_TESTS=1`.
- Always set a small `max_tokens`; disable qwen3 thinking. Never call Opus.

## 5. Feasibility & the "impractical" policy

No use case is strictly *impossible* in any approach — raw-api can do everything with enough hand-written plumbing, and langgraph can do everything natively. "Impractical" here means one of:

1. **Approach-impractical** — building it in this approach is so much DIY effort that the *real* lesson is "use a different approach" (e.g. multi-agent / HITL in raw-api). → We still ship a **minimal working demo** plus a `README.md` section "Why this is impractical here" pointing to the approach that does it cleanly.
2. **Local-model-impractical** — the pattern needs reliable tool-calling/structured output that `qwen3:1.7b`/`qwen2.5-coder:1.5b` don't deliver. → Default that project to a cloud model (documented) or `qwen2.5:7b`, and note the limitation.
3. **Approach-overkill** (new with claude-agent-sdk) — the implementation is short and correct, but the machinery earns little: no loop runs, or a single call would do the same job cheaper. → Ship it in full, and say so plainly in the README with a pointer to the simpler sibling. Applies to `claude-agent-sdk/03` and `/09`.

Every flagged combo is recorded in `TRACKING.md` and repeated in the folder's `README.md`. Nothing is silently dropped.

## 6. Per-use-case notes (deltas from the guide)

1. **RAG** — embeddings stay local via `sentence-transformers` (`all-MiniLM-L6-v2`) + a local vector store (Chroma/FAISS or pgvector on `aiup-postgres:15432`). No Anthropic needed.
2. **Code generation** — `qwen2.5-coder`; execute generated code in a subprocess sandbox with timeout; iterate on test failures (cap iterations).
3. **Data extraction** — Pydantic schema + JSON mode; validate & retry once. Heavier local model or Haiku if 1.5b JSON is flaky.
4. **Research agent** — pluggable search tool; if no internet, ship a local/mock corpus search so it runs air-gapped. Tool-calling reliability → Haiku/qwen2.5:7b.
5. **Support triage** — intent classifier + conditional routing + conversation memory (Valkey/in-memory). Fine on local Qwen.
6. **SQL/DB agent** — bundled SQLite sample DB; schema-injection prompt; validate SQL (read-only) before execute.
7. **Multi-agent** — langgraph native (sub-graphs); langchain workaround; raw-api minimal + impractical note.
8. **Autonomous ReAct** — langgraph native cycles; raw-api hand-written loop (educational); needs Haiku/7b for reliable tool selection.
9. **Recommendations** — profile store + retrieval + ranking + NL explanation. Fine on local Qwen.
10. **HITL approval** — langgraph `interrupt()` + checkpointer; raw-api manual pause/resume via DB + `/resume` (minimal + impractical note); langchain callback workaround.

## 7. claude-agent-sdk deltas (per use case)

The harness is the same everywhere; what changes is which SDK feature carries the use case.

| # | What the Agent SDK version does differently | Verdict |
|---|---|---|
| 1 | Retrieval is agent-driven `Grep`/`Glob`/`Read` — **no vector store, no embeddings** | works; lexical-only, so semantic misses are possible |
| 2 | Built-in `Write`/`Edit`/`Bash`: the agent writes code *and tests*, runs pytest, and fixes failures. **No loop in the repo.** | ⭐ showcase |
| 3 | Tool-as-schema: the record arrives as the tool call's *input*, validated with Pydantic | ⚠️ one-shot; harness earns little |
| 4 | Built-in `WebSearch`/`WebFetch`, **opt-in** (`RESEARCH_ALLOW_WEB`); offline corpus mode is the default so it runs air-gapped | works |
| 5 | Routing is agentic — the agent decides whether to call `lookup_order`; the response records whether it did | works |
| 6 | The agent discovers the schema via tools rather than it being prompt-injected; two independent read-only defences | works |
| 7 | `agents={}` subagents, each with **its own context and tool allow-list** (least privilege per role) | ⭐ showcase |
| 8 | The SDK *is* the ReAct loop — no text protocol to parse, immune to the `Action Input` redaction bug | ⭐ showcase |
| 9 | Profile + catalog fetched through tools; every recommended id validated against the real catalog | ⚠️ modest win at this scale |
| 10 | `can_use_tool` async permission callback — the agent parks itself mid-run awaiting a future | ⭐ showcase, but **in-process only**: not durable, single worker |
