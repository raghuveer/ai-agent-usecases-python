# Build Plan

Sequenced delivery of the 30 examples. Governed by `SPEC.md`; status tracked in `TRACKING.md`.

## Phase 0 — Foundations (do once, before any use case)
1. **Repo skeleton:** create `raw-api/`, `langchain/`, `langgraph/` use-case subfolders (`01-rag` … `10-hitl`). Add root `README.md` (overview + the approach-comparison table from the guide), `LICENSE`, `.gitignore` (`.env`, `__pycache__`, vectorstores, `*.db`).
2. **Shared template** under `_template/` per approach: `pyproject.toml`, `.env.example`, `settings.py` (pydantic-settings: `LLM_BASE_URL`, `LLM_GATEWAY_KEY`, `LLM_MODEL`), a thin `llm.py` client factory (OpenAI-compatible, injectable for tests), `app/main.py` (`/health` + `/run`), and `tests/` with a mocked-LLM unit test. Every project is copied from this.
3. **Verify the template** round-trips against local Qwen and (one gated test) Haiku.

## Phase 1 — Reference trio (prove the pattern end-to-end)
Build **UC1 RAG** in all three approaches first — it exercises config, the LLM client, embeddings, tests, and README structure. Use it as the canonical example all later folders are modeled on. Review, then lock conventions.

## Phase 2 — Local-friendly use cases (no Anthropic spend)
Batch the ones that run well on local Qwen, all three approaches each:
**UC2 code-gen, UC3 extraction, UC5 triage, UC6 SQL, UC9 recommendations.**

## Phase 3 — Tool-calling / agentic use cases (Haiku-backed, capped)
**UC4 research, UC8 ReAct** — default to Haiku or `qwen2.5:7b`; gate Anthropic tests.

## Phase 4 — Coordination / HITL (the approach-divergent ones)
**UC7 multi-agent, UC10 HITL** — full in langgraph, workaround in langchain, minimal-stub + impractical README in raw-api.

## Phase 5 — Polish for public release
- Root README with a navigable matrix linking every folder.
- Per-folder READMEs complete (run steps, env, trade-offs, feasibility notes).
- One CI workflow running **unit tests only** (no network) across all projects.
- `git init`, first commit, push to the public GitHub repo.

## Working rules
- One use case × approach at a time; copy from `_template/`, keep diffs reviewable.
- Update `TRACKING.md` build-state cell as each lands; record any feasibility re-rating.
- Default everything to local Qwen; only spend Anthropic budget where `TRACKING.md` says so, with `max_tokens` caps.

## Decisions (locked 2026-06-24)
- **Auth key:** per-project platform **virtual key** (issued via the platform), read from env `LLM_GATEWAY_KEY`. Master key only as a local fallback.
- **Vector store for RAG:** **Chroma** (file-based, portable) with local `sentence-transformers` embeddings.
- **Packaging:** **`pyproject.toml` + `uv`** for every project.

## Platform provisioning convention (project / phase / key)
Each use case maps to one AIUP **phase** (an engagement) under the **project** `ai-usecases` (org `acme-corp`). A single **virtual key** per phase — alias `enterprise_architect-poc-ai-usecases-uc<NN>-<slug>` — is reused by that use case's raw-api/langchain/langgraph integration tests. Key values live only in each project's gitignored `.env`. Apps call the gateway with an allow-listed alias: `qwen-local-instruct` (free default) / `qwen-local-coder` (free, code) / `claude-haiku-4-5` (budget). Seeding commands and ids: see memory `aiup-usecases-provisioning`.

Phases provisioned (one key each): uc01-rag, uc02-codegen, uc03-extraction, uc04-research, uc05-triage, uc06-sql, uc07-multiagent, uc08-react, uc09-recommendations, uc10-hitl — all 10.

## Current state
- **Phase 0** — per-approach `_template/` built; 30 folders scaffolded; root docs in place. ✅
- **Phase 1** — UC1 RAG trio built + unit/integration green on free local Qwen. ✅
- **Phase 2** — UC2, UC3, UC5, UC6, UC9 built across all three approaches; unit tests green, integration green via gateway. ✅
  - UC2/5/6/9 run on **free local Qwen** (`qwen-local-instruct` / `qwen-local-coder`).
  - **UC3 uses `claude-haiku-4-5`** — local models proved too unreliable for strict invoice JSON (the apparent "PII guardrail" was weak-model output, not redaction). Integration test marked `anthropic` (small capped spend); unit tests stay mocked/offline.
- **Phase 3** — UC4 research, UC8 ReAct built across all three approaches; unit green, integration green. Both default to `claude-haiku-4-5` (free local can't drive a text ReAct loop). ✅
- **Phase 4** — UC7 multi-agent (Haiku), UC10 HITL (free local) built across all three; unit + integration green. raw-api/07 and raw-api/10 carry "why impractical here" notes pointing to their langgraph siblings. ✅
- **ALL 10 use cases × 3 approaches = 30 projects complete.**
- **Remaining: Phase 5 (release)** — root README navigable matrix, a CI workflow running unit tests only (no network) across all projects, final per-folder README pass, `git init` + first commit + push.
