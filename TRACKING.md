# Feasibility & Build Tracking

Status of each use case × approach. Legend:
- ✅ **Build** — natural fit, ship full example on local Qwen.
- ⚠️ **Build w/ caveat** — works, but awkward in this approach and/or needs Haiku / `qwen2.5:7b` for reliability (noted).
- 📝 **Impractical-but-stub** — ship a *minimal* demo + `README.md` "Why impractical here" pointing to the clean approach. Tracked, never dropped.

Build state column: ☐ not started · ◐ in progress · ☑ done.

| # | Use case | raw-api | langchain | langgraph | claude-agent-sdk | Model default | Build state |
|---|----------|:------:|:--------:|:--------:|:----------------:|---------------|:----------:|
| 1 | Q&A / RAG chatbot | ✅ | ✅ | ⚠️ overkill | ⚠️ no vector store — lexical only | local Qwen + local embeddings · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 2 | Code generation | ✅ | ✅ | ⚠️ only if iterative | ⭐ native (Write/Bash loop) | `qwen-local-coder` · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 3 | Data extraction | ✅ | ✅ | ⚠️ unless pipeline | ⚠️ one-shot; harness earns little | **cloud** (local JSON too flaky) | ☑ ☑ ☑ ☑ |
| 4 | Research agent | ⚠️ manual loop | ✅ | ✅ | ✅ built-in WebSearch/WebFetch | **cloud** (local fails ReAct) | ☑ ☑ ☑ ☑ |
| 5 | Customer support triage | ⚠️ no native routing | ✅ | ✅ | ✅ agentic routing | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 6 | SQL / DB agent | ✅ | ✅ | ⚠️ with validators | ✅ agent discovers schema | `qwen-local-coder` · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 7 | Multi-agent orchestration | 📝 complex DIY | ⚠️ workaround | ✅ native | ⭐ native subagents + per-role tools | **cloud** | ☑ ☑ ☑ ☑ |
| 8 | Autonomous ReAct | ⚠️ hand-written loop | ⚠️ AgentExecutor | ✅ graph cycles | ⭐ the SDK *is* the loop | **cloud** (local fails ReAct) | ☑ ☑ ☑ ☑ |
| 9 | Recommendations | ✅ | ✅ | ⚠️ profile+state | ⚠️ modest win over one call | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 10 | Human-in-the-loop approval | 📝 manual queue | ⚠️ callback workaround | ✅ `interrupt()` (durable) | ⭐ `can_use_tool` (in-process only) | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |

Build-state cells map left→right to raw-api / langchain / langgraph / claude-agent-sdk.
⭐ = the approach's showcase for that use case.

## Combos flagged impractical (carry a README note) — ALL BUILT ✅
- **raw-api/07-multi-agent** — hand-rolled orchestrator; "Why impractical in raw API" note → `langgraph/07`. Built + green.
- **raw-api/10-hitl-approval** — hand-built checkpoint store + `/resume`; note → `langgraph/10` (`interrupt()`). Built + green.
- **claude-agent-sdk/03-data-extraction** — an agent harness doing a one-shot job; no loop runs, no built-in tools used. Note → `raw-api/03` (one HTTP call, cheaper per document).
- **claude-agent-sdk/09-recommendations** — ranking 7 items needs no loop; the win (grounded explanations, partial catalog reads) is modest at this scale. Note → `raw-api/09`.
- **claude-agent-sdk/01-rag** — no vector store: retrieval is lexical `Grep`/`Read`, so semantically-phrased questions can miss. Note → `langchain/01` for semantic recall.

## Empirical model findings (after building)
- **Local Qwen handles:** RAG (1), code-gen (2), triage (5), SQL (6), recommendations (9), and HITL drafting (10) — all on free `qwen-local-instruct`/`qwen-local-coder`.
- **Needs a cloud model:** data-extraction (3, strict JSON), research (4), ReAct (8), multi-agent (7) — local models can't reliably do strict-JSON or multi-step text ReAct. These integration tests are marked `anthropic` (small capped spend).
- **The whole `claude-agent-sdk` column needs a cloud model** — even the six use cases the other approaches run free. The SDK drives the Claude Code harness (large system prompt + built-in tool loop); `qwen3:0.6b` and its larger local siblings cannot sustain it. Every project there caps `AGENT_MAX_TURNS` and `AGENT_MAX_BUDGET_USD`; `claude-sonnet` is the documented escalation for code-gen and multi-agent.
- **Structured tool calls sidestep the PII-redaction bug.** The gateway's Presidio redaction masked the literal ReAct label `Action Input` → `<PERSON>`, breaking text-ReAct parsing (mitigated there by renaming the field to `Arguments:`). `claude-agent-sdk/08` is immune: its tool calls are structured protocol messages, so there is no prose control protocol to mangle.
- See memory `aiup-gateway-quirks` for the redaction behavior and ReAct stop-sequence mitigations.

## Platform drift (found 2026-07-29)
- Gateway moved **`:8080` → `:8094`**; LiteLLM tier removed (ADR 0036) — `projects-ms` mints native `sk-aiup-…` keys, and model aliases dropped version suffixes (`claude-haiku`, not `claude-haiku-4-5`).
- **Consequence:** the 30 pre-existing projects' `.env` / `.env.example` point at the retired port with old aliases, so their integration tests need a one-line config update before they run. `claude-agent-sdk/` targets the new endpoint.

## Live-run findings (claude-agent-sdk, validated 2026-07-29)

Three use cases were run live against the gateway (UC02, UC08, UC10). **Every one of the following was invisible to the mocked unit tests and only surfaced under a real agent** — the clearest argument in this repo for keeping gated integration tests:

1. **`can_use_tool` requires streaming input.** Passing a string prompt while a permission callback is set raises `ValueError: can_use_tool callback requires streaming mode`. `default_runner` now wraps the prompt in an `AsyncIterable` when a gate is present.
2. **`allowed_tools` silently shadows the permission gate.** Listing the guarded tool there auto-approves it *before* `can_use_tool` is consulted — the agent sent the message with no approval. The SDK emits `CanUseToolShadowedWarning`. UC10 now passes `allowed_tools=[]`; the tool is still available via its MCP server (that list controls auto-approval, not availability).
3. **Turn/budget caps raise, they do not return.** Exhausting `max_turns` / `max_budget_usd` raises from `query()` instead of yielding a `ResultMessage`, so `stop_reason` handling never ran. `default_runner` maps those two messages back to `stop_reason="max_turns"` / `"max_budget"`; genuine errors still propagate.
4. **A denial with `interrupt=True` surfaces as an error result.** For UC10 that is the expected terminal state, so `resolve_run` translates it into a rejection — but only on the denied path.
5. **`cwd` is not a sandbox.** The `Write` tool accepts absolute paths, and the model repeatedly wrote to `/tmp/solution.py` instead of the workdir, making UC02 flaky. Mitigated by an explicit relative-paths-only instruction plus reading artefacts back only from the workdir. Not a security boundary — a container/VM or the SDK `sandbox` setting is.
6. **Prompt caching does not pass through the gateway** (`cache_read_input_tokens: 0`), so every agent turn re-pays the full Claude Code harness prompt. Measured: ~847 input tokens for a *trivial* one-shot call, and **$0.35–0.48 for a 9–10 turn code-gen run** on `claude-haiku`. The original `$0.25` cap was exhausted in 15 seconds. Defaults raised to `AGENT_MAX_TURNS=12`, `AGENT_MAX_BUDGET_USD=1.00`.
7. **The gateway's `empty_input` guardrail blocks short prompts** — a `"Say OK"` probe returns `400 NO_SOURCE_PROVIDED / empty_input`. Affects hand-testing with curl, not the examples themselves.

## Status
- **10/10 use cases × 4 approaches = 40 projects built.**
- `claude-agent-sdk`: **131 offline unit tests green** (up from 125 — six added as regressions for the bugs above). 14 integration tests written and gated; **UC02, UC08, UC10 verified live and passing**. The remaining seven are written and collecting but not yet run live.
- raw-api / langchain / langgraph: unit + integration green as of v0.2.0 — but see Platform drift: their `.env` still points at the retired `:8080`.
