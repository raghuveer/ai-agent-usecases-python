# Feasibility & Build Tracking

Status of each use case × approach. Legend:
- ✅ **Build** — natural fit, ship full example on local Qwen.
- ⚠️ **Build w/ caveat** — works, but awkward in this approach and/or needs Haiku / `qwen2.5:7b` for reliability (noted).
- 📝 **Impractical-but-stub** — ship a *minimal* demo + `README.md` "Why impractical here" pointing to the clean approach. Tracked, never dropped.

Build state column: ☐ not started · ◐ in progress · ☑ done.

| # | Use case | raw-api | langchain | langgraph | Model default | Build state |
|---|----------|:------:|:--------:|:--------:|---------------|:----------:|
| 1 | Q&A / RAG chatbot | ✅ | ✅ | ⚠️ overkill | local Qwen + local embeddings | ☑ ☑ ☑ |
| 2 | Code generation | ✅ | ✅ | ⚠️ only if iterative | `qwen-local-coder` | ☑ ☑ ☑ |
| 3 | Data extraction | ✅ | ✅ | ⚠️ unless pipeline | **claude-haiku-4-5** (local JSON too flaky) | ☑ ☑ ☑ |
| 4 | Research agent | ⚠️ manual loop | ✅ | ✅ | **claude-haiku-4-5** (local fails ReAct) | ☑ ☑ ☑ |
| 5 | Customer support triage | ⚠️ no native routing | ✅ | ✅ | local Qwen | ☑ ☑ ☑ |
| 6 | SQL / DB agent | ✅ | ✅ | ⚠️ with validators | `qwen-local-coder` | ☑ ☑ ☑ |
| 7 | Multi-agent orchestration | 📝 complex DIY | ⚠️ workaround | ✅ native | **claude-haiku-4-5** | ☑ ☑ ☑ |
| 8 | Autonomous ReAct | ⚠️ hand-written loop | ⚠️ AgentExecutor | ✅ graph cycles | **claude-haiku-4-5** (local fails ReAct) | ☑ ☑ ☑ |
| 9 | Recommendations | ✅ | ✅ | ⚠️ profile+state | local Qwen | ☑ ☑ ☑ |
| 10 | Human-in-the-loop approval | 📝 manual queue | ⚠️ callback workaround | ✅ `interrupt()` | local Qwen | ☑ ☑ ☑ |

Build-state cells map left→right to raw-api / langchain / langgraph.

## Combos flagged impractical (carry a README note) — ALL BUILT ✅
- **raw-api/07-multi-agent** — hand-rolled orchestrator; "Why impractical in raw API" note → `langgraph/07`. Built + green.
- **raw-api/10-hitl-approval** — hand-built checkpoint store + `/resume`; note → `langgraph/10` (`interrupt()`). Built + green.

## Empirical model findings (after building)
- **Local Qwen handles:** RAG (1), code-gen (2), triage (5), SQL (6), recommendations (9), and HITL drafting (10) — all on free `qwen-local-instruct`/`qwen-local-coder`.
- **Needs `claude-haiku-4-5`:** data-extraction (3, strict JSON), research (4), ReAct (8), multi-agent (7) — local models can't reliably do strict-JSON or multi-step text ReAct. These integration tests are marked `anthropic` (small capped spend).
- See memory `aiup-gateway-quirks` for the PII-redaction behavior and ReAct stop-sequence mitigations.

## Status: 10/10 use cases × 3 approaches = 30 projects built, unit + integration green.
