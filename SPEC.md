# Specification — 10 Agent Use Cases × 3 Approaches

This spec governs the 30 example projects (`<approach>/<usecase>/`). Read alongside `agent-development-guide.md` (use-case definitions), `CLAUDE.md` (platform access + cost rules), `PLAN.md` (build order), and `TRACKING.md` (feasibility matrix).

## 1. Goals & non-goals

**Goals:** a public, clonable reference where each use case is implemented in raw-api, langchain, and langgraph so readers can *compare* the three approaches; each project independently runnable with its own tests; cost kept near-zero by defaulting to self-hosted Qwen.

**Non-goals:** production hardening, auth/multi-tenancy, polished UIs, exhaustive provider coverage. Examples optimise for **clarity of the approach**, not feature completeness.

**Dev vs runtime (do not conflate):** *Authoring* this code is done by Claude Code on the maintainer's Claude Max subscription — it never touches the platform's Anthropic budget. The **AI Utility Platform is solely the runtime LLM target the example apps call** to integrate and test approach/usecase behavior. So the budget rules below constrain the *apps' calls*, not the development process.

## 2. Model strategy

| Tier | Model | Use it for |
|---|---|---|
| Default (local, free) | `qwen3:1.7b` | general chat/reasoning, classification, routing |
| Local code | `qwen2.5-coder:1.5b` | code generation, SQL |
| Local tiny | `qwen3:0.6b` | smoke tests, latency demos |
| Local heavy | `qwen2.5:7b-instruct` / `qwen2.5-coder:7b-instruct` | when 1.5–1.7b tool-calling/JSON is too unreliable but you want to stay local |
| Cloud default | `claude-haiku-4-5` | the few cases where small local models can't deliver reliable tool-calls / structured output |
| Cloud opt-in | `claude-sonnet-4-6` | hardest reasoning/multi-agent only, behind an env flag |
| Excluded | Opus | budget — do not use |

Model id is **always** resolved from env (`LLM_MODEL`), with a per-project sensible default. Apps never hardcode a provider — they speak OpenAI-compatible HTTP to the gateway (`LLM_BASE_URL`, default `http://localhost:8080`).

### Framework wiring
- **raw-api:** `httpx`/`openai` SDK pointed at `LLM_BASE_URL`. The agent loop, memory, and state are hand-written (this is the point).
- **langchain / langgraph:** use `langchain-openai`'s `ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_GATEWAY_KEY, model=LLM_MODEL)` for **both** Qwen and Claude — one uniform client path keeps the comparison honest and avoids per-provider branching.

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
- **Determinism for tests:** the LLM client is injected/overridable so unit tests swap in a stub.

## 4. Testing & cost rules

- **Unit:** mock the model client; assert routing/parsing/loop logic. Zero network. These run in CI.
- **Integration:** default `LLM_MODEL` = local Qwen; assert the app round-trips against the live local model. Mark Anthropic-requiring tests with `@pytest.mark.anthropic` and skip unless `RUN_ANTHROPIC_TESTS=1`.
- Always set a small `max_tokens`; disable qwen3 thinking. Never call Opus.

## 5. Feasibility & the "impractical" policy

No use case is strictly *impossible* in any approach — raw-api can do everything with enough hand-written plumbing, and langgraph can do everything natively. "Impractical" here means one of:

1. **Approach-impractical** — building it in this approach is so much DIY effort that the *real* lesson is "use a different approach" (e.g. multi-agent / HITL in raw-api). → We still ship a **minimal working demo** plus a `README.md` section "Why this is impractical here" pointing to the approach that does it cleanly.
2. **Local-model-impractical** — the pattern needs reliable tool-calling/structured output that `qwen3:1.7b`/`qwen2.5-coder:1.5b` don't deliver. → Default that project to `claude-haiku-4-5` (documented) or `qwen2.5:7b`, and note the limitation.

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
