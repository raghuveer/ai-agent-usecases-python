# Build Plan

Sequenced delivery of the 40 examples. Governed by `SPEC.md`; status tracked in `TRACKING.md`.

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
Each use case maps to one AIUP **phase** (an engagement) under the **project** `ai-usecases` (org `acme-corp`). A single **virtual key** per phase — alias `enterprise_architect-poc-ai-usecases-uc<NN>-<slug>` — is reused by that use case's raw-api/langchain/langgraph integration tests. Key values live only in each project's gitignored `.env`. Apps call the gateway with an allow-listed alias: `qwen-local-instruct` (free default) / `qwen-local-coder` (free, code) / `claude-haiku` (budget). Seeding commands and ids: see memory `aiup-usecases-provisioning`.

Phases provisioned (one key each): uc01-rag … uc10-hitl — all 10. **Superseded 2026-07-29:** the platform rebuild wiped the `ai-usecases` project and every one of those keys. A single phase `poc-ai-usecases-agentsdk` was recreated with one key (aliases `claude-haiku`, `claude-sonnet`, `qwen-local-instruct`, `qwen-local-coder`) and is currently shared by all 40 projects. Re-provisioning per-use-case phases is a tidy-up task, not a blocker.

## Phase 6 — 4th approach: `claude-agent-sdk` (2026-07-29)

Added the Python **Claude Agent SDK** (`claude-agent-sdk`) as a fourth approach, all 10 use cases.

Build order (hardest-first, to de-risk the shared seam early):
1. `_template` — established the three reusable pieces: `sdk_env()` config translation, the injectable `Runner` seam, and `build_options()` budget discipline. Validated before replicating.
2. **UC10 HITL** first — the hardest pattern (park/resume through `can_use_tool`). Building it first surfaced that a parked coroutine needs one long-lived event loop, which shaped both the design and the tests.
3. Remaining showcases: UC02 code-gen, UC07 multi-agent, UC08 ReAct.
4. The other six: UC01, UC03, UC04, UC05, UC06, UC09.

Working rules that differ from Phases 0–5:
- **No free-local fallback.** Unlike the other three approaches, every project here defaults to a cloud model; `max_turns` + `max_budget_usd` are mandatory on every run, and all integration tests are double-gated (`RUN_INTEGRATION=1` + `RUN_ANTHROPIC_TESTS=1`).
- **Mock at the runner, not the client.** `query()` spawns the Claude Code CLI, so unit tests inject a `runner` stub; `collect()` is tested separately against real SDK message types.
- **Node.js + Claude Code CLI** are a live-run prerequisite (the Python SDK spawns the CLI). Unit tests need neither, so CI is unaffected.

## Current state
- **Phase 0** — per-approach `_template/` built; 30 folders scaffolded; root docs in place. ✅
- **Phase 1** — UC1 RAG trio built + unit/integration green on free local Qwen. ✅
- **Phase 2** — UC2, UC3, UC5, UC6, UC9 built across all three approaches; unit tests green, integration green via gateway. ✅
  - UC2/5/6/9 run on **free local Qwen** (`qwen-local-instruct` / `qwen-local-coder`).
  - **UC3 uses `claude-haiku-4-5`** — local models proved too unreliable for strict invoice JSON (the apparent "PII guardrail" was weak-model output, not redaction). Integration test marked `anthropic` (small capped spend); unit tests stay mocked/offline.
- **Phase 3** — UC4 research, UC8 ReAct built across all three approaches; unit green, integration green. Both default to `claude-haiku-4-5` (free local can't drive a text ReAct loop). ✅
- **Phase 4** — UC7 multi-agent (Haiku), UC10 HITL (free local) built across all three; unit + integration green. raw-api/07 and raw-api/10 carry "why impractical here" notes pointing to their langgraph siblings. ✅
- **Phase 5** — released as v0.2.0: navigable root README, CI running unit tests only, per-folder READMEs, security review, dependency gates. ✅
- **Phase 6** — `claude-agent-sdk` × 10 use cases built; **131 offline unit tests green**; 14 integration tests written, collecting, and gated. ✅
- **Phase 6b — live validation** (2026-07-29): fresh `sk-aiup-…` key minted under phase `poc-ai-usecases-agentsdk`; **all 10 use cases run live and passing** across two passes. Exposed **8 defects invisible to mocked tests** (see `TRACKING.md` → Live-run findings) — including `setting_sources=[]` failing to isolate developer memory, and the delegation tool being named `Agent` rather than `Task`. All fixed with regression tests; turn/budget defaults re-baselined from measured cost. ✅
- **ALL 10 use cases × 4 approaches = 40 projects complete.**

### Notes for whoever runs this next
- **Minting keys now needs MFA.** `seed-virtual-keys.mjs` predates ADR 0042: a bare login returns a `token_use: mfa_pending` token (300s TTL) with no roles, so project creation 403s with "admin role required". Complete the step-up at `POST /v1/auth/mfa/verify` with `{"factorType": "...", "code": "..."}` before driving projects-ms.
- ~~Re-point the 30 older projects~~ — **done 2026-07-29**: 66 files swept to `:8094` + unsuffixed aliases, all 30 `.env` re-keyed, live-verified across all three approaches (350 offline tests green). See `TRACKING.md` → Platform drift.
