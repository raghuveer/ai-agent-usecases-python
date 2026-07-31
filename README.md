# AI Agent Use Cases — Raw API · LangChain · LangGraph · Claude Agent SDK

[![unit-tests](https://github.com/raghuveer/ai-agent-usecases-python/actions/workflows/ci.yml/badge.svg)](https://github.com/raghuveer/ai-agent-usecases-python/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/raghuveer/ai-agent-usecases-python?label=release)](https://github.com/raghuveer/ai-agent-usecases-python/releases)
[![security review](https://img.shields.io/badge/security-reviewed-brightgreen)](docs/security-review.md)
[![dependencies: pip-audit](https://img.shields.io/badge/deps-pip--audit%20%2B%20Dependabot-blue)](.github/dependabot.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue)](#)

> Ten common LLM agent use cases — RAG, code-gen, data extraction, research, support triage, SQL, multi-agent, ReAct, recommendations, human-in-the-loop — each implemented four ways: **raw API**, **LangChain**, **LangGraph**, and the **Claude Agent SDK**. Python + FastAPI, with unit & integration tests. A practical side-by-side of the four approaches.

Ten common agent use cases, each implemented **four ways** — direct **Raw API**, **LangChain**, **LangGraph**, and the **Claude Agent SDK** — in **Python + FastAPI**. The point is to *compare* the approaches on the same problems. Every folder is a self-contained, runnable project with its own tests.

**Other languages:** a TypeScript edition (LangChain.js + LangGraph.js) is planned at `ai-agent-usecases-typescript` _(coming soon)_.

> **Status — v0.7.1:** all 10 use cases built across all 4 approaches (40 projects + a `_template/` per approach), each with offline, mocked **unit tests** and gated **integration tests**. **602 unit tests green repo-wide, and all 40 projects verified live** — then re-verified against the committed lockfiles, so what you install is what was tested. Run any of it with **`docker compose up`** (no key, no account), see what an agent actually did with **`?trace=1`**, and watch it work with **`POST /run/stream`** — both on the three use cases where an agent loops, delegates or pauses (UC07, UC08, UC10, all four approaches). Live runs have caught **17 bugs the mocked tests could not see** — a permission gate that failed *open*; `setting_sources=[]` silently failing to stop the developer's private `~/.claude` memory leaking into agent context; a capped agent run discarding everything it had already produced; and three separate gateway limitations (`stop` arrays, streaming, prompt caching). All fixed or documented, each with a regression test. See `TRACKING.md` → Live-run findings. **Security-reviewed** against NIST · OWASP LLM Top 10 · OWASP Web Top 10 (see [`docs/security-review.md`](docs/security-review.md) / [`SECURITY.md`](SECURITY.md)); dependencies patched to current majors, with a CI `pip-audit` gate and Dependabot keeping them current. Examples optimise for clarity of the approach, not production hardening.

> 🔴 **Read [`docs/security-review.md` §11](docs/security-review.md) before deploying anything from `claude-agent-sdk/`.** That approach has a materially larger blast radius than the other three: `02-code-generation` grants the agent a **shell by default** (remote code execution by design, driven by untrusted request text), and `cwd` was empirically shown **not** to confine file writes. Bandit and pip-audit are clean on it (0 High/Medium, no known CVEs), but F9 is rated **High as shipped** and needs a real sandbox — container/VM, no network, resource caps.

## Use-case matrix

| # | Use case | raw-api | langchain | langgraph | claude-agent-sdk | Runtime model |
|---|----------|:-------:|:---------:|:---------:|:----------------:|---------------|
| 1 | Q&A / RAG chatbot | [▸](raw-api/01-rag) | [▸](langchain/01-rag) | [▸](langgraph/01-rag) | [▸](claude-agent-sdk/01-rag) | local Qwen · Haiku² |
| 2 | Code generation | [▸](raw-api/02-code-generation) | [▸](langchain/02-code-generation) | [▸](langgraph/02-code-generation) | [★](claude-agent-sdk/02-code-generation) | local Qwen (coder) · Haiku² |
| 3 | Data extraction | [▸](raw-api/03-data-extraction) | [▸](langchain/03-data-extraction) | [▸](langgraph/03-data-extraction) | [▸](claude-agent-sdk/03-data-extraction) | Haiku¹ |
| 4 | Research agent | [▸](raw-api/04-research-agent) | [▸](langchain/04-research-agent) | [▸](langgraph/04-research-agent) | [▸](claude-agent-sdk/04-research-agent) | Haiku¹ |
| 5 | Customer support triage | [▸](raw-api/05-support-triage) | [▸](langchain/05-support-triage) | [▸](langgraph/05-support-triage) | [▸](claude-agent-sdk/05-support-triage) | local Qwen · Haiku² |
| 6 | SQL / DB agent | [▸](raw-api/06-sql-agent) | [▸](langchain/06-sql-agent) | [▸](langgraph/06-sql-agent) | [▸](claude-agent-sdk/06-sql-agent) | local Qwen (coder) · Haiku² |
| 7 | Multi-agent orchestration | [▸](raw-api/07-multi-agent) | [▸](langchain/07-multi-agent) | [▸](langgraph/07-multi-agent) | [★](claude-agent-sdk/07-multi-agent) | Haiku¹ |
| 8 | Autonomous ReAct | [▸](raw-api/08-autonomous-react) | [▸](langchain/08-autonomous-react) | [▸](langgraph/08-autonomous-react) | [★](claude-agent-sdk/08-autonomous-react) | Haiku¹ |
| 9 | Recommendations | [▸](raw-api/09-recommendations) | [▸](langchain/09-recommendations) | [▸](langgraph/09-recommendations) | [▸](claude-agent-sdk/09-recommendations) | local Qwen · Haiku² |
| 10 | Human-in-the-loop approval | [▸](raw-api/10-hitl-approval) | [▸](langchain/10-hitl-approval) | [▸](langgraph/10-hitl-approval) | [★](claude-agent-sdk/10-hitl-approval) | local Qwen · Haiku² |

★ = showcase for that approach.

¹ **Why Haiku for 3, 4, 7, 8?** Building these we found small local models can't reliably emit strict schema JSON (extraction) or drive a multi-step text **ReAct** loop (research / ReAct / multi-agent) — they wandered into prose and skipped tools. Those four default to a cloud model; the rest run on free self-hosted Qwen. See `TRACKING.md` for the full feasibility matrix and `SPEC.md` for the cost rules.

² **The `claude-agent-sdk` column always uses a cloud model**, even for the six use cases the other approaches run free on local Qwen. That is not a preference — the SDK drives the Claude Code harness (large system prompt + built-in tool loop), which small local models cannot sustain. Budget your runs accordingly: every project caps `AGENT_MAX_TURNS` and `AGENT_MAX_BUDGET_USD`.

> **Measured cost, so you can plan.** Prompt caching does **not** pass through the local gateway (`cache_read_input_tokens: 0`), so every agent turn re-pays the full harness prompt. A 9–10 turn code-generation run on `claude-haiku` cost **$0.35–0.48**. Defaults are `AGENT_MAX_TURNS=12` / `AGENT_MAX_BUDGET_USD=1.00`; an earlier `$0.25` cap was exhausted in 15 seconds. Going direct to `api.anthropic.com` (drop `LLM_BASE_URL`) restores caching and costs materially less.

Which approach suits which use case (and the two raw-api combos deliberately kept minimal — multi-agent and HITL — with "why impractical here" notes) is tracked in **`TRACKING.md`**.

### See the four approaches side by side

**[`docs/compare/`](docs/compare/README.md)** — one page per use case: the core of
each of the four implementations, how much code each costs, and (for
[UC08](docs/compare/08-autonomous-react.md)) what a *real traced run* of each
measured. Generated from the source it describes (`scripts/compare_usecase.py`),
and CI fails if it drifts.

Two things that page settles with numbers rather than opinion:

- **raw-api, LangChain and LangGraph send byte-identical payloads** — 3,421 input
  / 125 output tokens each. The frameworks add no prompt overhead here.
- **Visibility falls as the framework writes more of your loop.** Add `?trace=1`
  to any of those four projects and you get the exact messages sent, every tool
  call and result, latency and tokens — except on the Agent SDK, which owns the
  loop and so cannot report most of it. It reports what the others cannot: real
  cost. See [`docs/trace-format.md`](docs/trace-format.md).

## The four approaches at a glance

| | You write | Best at | Weakest at |
|---|---|---|---|
| **raw-api** | every byte sent to the model, the loop, the memory | seeing exactly what happens | coordination-heavy work (7, 10 are minimal + "impractical here" notes) |
| **langchain** | chains, retrievers, tool wrappers | fast linear prototypes | cycles and conditional routing |
| **langgraph** | a typed state graph | cycles, conditional edges, **durable** pause/resume | flat one-shot tasks (structural cost with no payoff) |
| **claude-agent-sdk** | tools + a prompt; the SDK owns the loop | agentic work — built-in file/shell tools, subagents, permission gates | one-shot tasks (3, 9) where no loop is needed, and anything needing free local models |

**Where the Agent SDK genuinely wins** (its four ★ showcases):

- **02 code-generation** — built-in `Write`/`Bash` mean the agent writes code, writes tests, *runs* them, and fixes failures. There is no loop in the repo.
- **07 multi-agent** — subagents are a dict of `AgentDefinition`s, each with its **own context and tool allow-list**. Least privilege per role, declared as data.
- **08 autonomous-react** — the SDK *is* the ReAct loop. No `Thought:`/`Action:` parser, no `stop=["Observation:"]`, and immune to the PII-redaction bug that broke text-ReAct parsing (see `TRACKING.md`).
- **10 hitl-approval** — `can_use_tool` is an async callback, so "ask a human" is just awaiting a future. The agent parks itself mid-run. *But* the paused state is a live coroutine — single process, lost on restart. For durable approvals, `langgraph/10`'s checkpointer still wins.

## How models are accessed

No provider is hardcoded anywhere — configure via env; each project ships a `.env.example`. The three original approaches speak the **OpenAI** surface; `claude-agent-sdk` speaks the **Anthropic** surface, which changes two things:

| Var | raw-api / langchain / langgraph | claude-agent-sdk |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094/v1` | `http://localhost:8094` — **no `/v1`** (the SDK appends `/v1/messages`) |
| `LLM_GATEWAY_KEY` | `sk-aiup-…` virtual key | same key, but exported as `ANTHROPIC_AUTH_TOKEN` |
| `LLM_MODEL` | `qwen-local-instruct` · `qwen-local-coder` · `claude-haiku` | `claude-haiku` · `claude-sonnet` (no local aliases — see ² above) |

Copy `.env.example` → `.env` (gitignored) and fill in your key. The `.env` is **never committed**; only `.env.example` is.

> **Auth gotcha (claude-agent-sdk):** the gateway requires `Authorization: Bearer` and rejects `x-api-key`. The SDK sends `Bearer` only when `ANTHROPIC_AUTH_TOKEN` is set — setting `ANTHROPIC_API_KEY` instead produces a 401. Each project's `app/agent.py` sets the right one.

> ℹ️ **Platform note.** The AI Utility Platform gateway listens on **`:8094`**, and the LiteLLM tier has been removed (ADR 0036) — `projects-ms` mints native `sk-aiup-…` virtual keys directly, and model aliases carry no version suffix (`claude-haiku`, not `claude-haiku-4-5`). **All 40 projects target this**; the older 30 were migrated on 2026-07-29 and re-verified live across all three approaches.

## Use a different model or provider

Nothing is hardcoded to a provider — every project speaks **OpenAI-compatible HTTP** and reads its model/endpoint from env. To switch, edit `.env` only — **no code changes**:

| To use… | `LLM_BASE_URL` | `LLM_MODEL` | Notes |
|---|---|---|---|
| Bundled gateway (default) | `http://localhost:8094/v1` | `qwen-local-instruct` · `claude-haiku` | virtual key in `LLM_GATEWAY_KEY` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | `LLM_GATEWAY_KEY` = your OpenAI key |
| Ollama (direct, local) | `http://localhost:11434/v1` | `qwen2.5:7b-instruct` | any non-empty key works |
| Together · Groq · OpenRouter · Azure OpenAI | their `/v1` URL | their model id | all OpenAI-compatible |
| Anthropic · Bedrock · Vertex | an OpenAI-compatible proxy `/v1` | the proxy's alias | the proxy normalises these onto the OpenAI surface |

**This table applies to the three OpenAI-surface approaches.** `claude-agent-sdk` is not provider-portable in the same way: it speaks the Anthropic Messages API, so it runs against the bundled gateway, `api.anthropic.com` directly (drop `LLM_BASE_URL`), or Bedrock/Vertex via the SDK's own env vars — but not against OpenAI or a bare Ollama. That is a real limitation of the approach, not of the wiring.

**Tuning knobs** (also env, also no code change): `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.

Two caveats that are about the *model*, not the code:
- **Capability ≠ code:** small local models can't reliably do strict-JSON extraction or multi-step ReAct — that's why use cases 3/4/7/8 default to a stronger model. Point them at a capable model and they work as-is; point them at a weak one and the code is unchanged but quality drops.
- **Model quirks live in one place** — `model_profile()` in each project's `app/llm.py` (e.g. disabling qwen3's "thinking" mode). Add a new model family there in a single spot.

> **Structured-output modes (UC3 prototype):** `03-data-extraction` also honours `LLM_STRUCTURED_MODE=text|native` — `text` is the portable prompt-and-parse path; `native` uses the provider's JSON / structured-output feature for higher reliability where available. See that project's README.

## Quick start — Docker (nothing to install, no API key)

```bash
docker compose up
# → http://localhost:8000/health
curl -X POST localhost:8000/run -H 'content-type: application/json' \
     -d '{"question":"How long is the return window?"}'
```

That brings up Ollama, pulls a small local model, and serves `raw-api/01-rag`
against it. No account, no key, no gateway.

> **First run downloads ~1.5 GB and takes several minutes** — 1.4 GB for the
> model, plus ~80 MB for the embedding model Chroma fetches on first start. Both
> are cached afterwards, so later runs start in seconds. `/health` answers only
> once the index is built; that is the wait.

Pick a different example with `PROJECT`:

```bash
PROJECT=langgraph/10-hitl-approval docker compose up --build
```

To see exactly what an agent did — the messages sent, every tool call, tokens and
latency — add `?trace=1`. That is implemented on the **UC08** projects today
(`raw-api`, `langchain`, `langgraph`, `claude-agent-sdk`), not yet on the rest;
see [`docs/trace-format.md`](docs/trace-format.md) and the
[UC08 comparison](docs/compare/08-autonomous-react.md).

> Covers the three OpenAI-surface approaches. **`claude-agent-sdk` is not in the
> Docker path**: it speaks the Anthropic Messages API and spawns the Claude Code
> CLI, so it needs Node plus a real Anthropic-compatible endpoint. Run those
> projects directly — each README explains the prerequisites.

**Already running Ollama locally?** The compose file deliberately does *not*
publish port 11434, so it will not collide with yours. Its Ollama is reachable
only from the app container.

**Rancher Desktop in containerd mode** has no Docker daemon — use `nerdctl
compose up` instead. It works, with one caveat: nerdctl ignores
`depends_on: condition`, so the app starts before the model finishes pulling and
its first request may fail. Wait for the pull, then retry. Real `docker compose`
gates this correctly.

## Quick start — local (any project)

Each project is independent and uses [`uv`](https://docs.astral.sh/uv/):

```bash
cd raw-api/01-rag                 # pick any of the 40
cp .env.example .env              # then edit .env: set LLM_GATEWAY_KEY (and LLM_MODEL if needed)

uv sync --extra dev                 # creates .venv, installs from uv.lock

# Offline unit tests — mocked LLM, no network, no key needed:
uv run pytest -q -m "not integration"

# Live integration test (needs the gateway + a real key in .env):
RUN_INTEGRATION=1 uv run pytest -q -m integration

# Serve the API:
uv run uvicorn app.main:app --reload
# → GET /health, POST /run   (see the project README for the request shape)
```

> **Every project commits a `uv.lock`**, so `uv sync` reproduces the exact dependency
> set the tests were run against rather than re-resolving whatever is newest. CI installs
> with `uv sync --locked`, which fails the build if a `pyproject.toml` is edited without
> re-locking — so the lockfiles cannot silently drift. After changing dependencies, run
> `uv lock` in that project and commit the result.

## Testing model

- **Unit tests** — the LLM is mocked; fully offline; these run in CI.
- **Integration tests** — call the live gateway; gated behind `RUN_INTEGRATION=1`. The four Haiku use cases also mark theirs `anthropic` since they spend a small, capped budget.

For `claude-agent-sdk`, the mocking seam is a **`runner` callable** injected into `create_app()` instead of an LLM client — `query()` spawns the Claude Code CLI, which unit tests must never do. The message-parsing layer (`collect()`) is tested separately against **real** `AssistantMessage`/`ResultMessage` objects, so parsing is still verified against the types the CLI actually emits. All of its integration tests are double-gated (`RUN_INTEGRATION=1` **and** `RUN_ANTHROPIC_TESTS=1`), because that approach has no free-local fallback.

CI (`.github/workflows/ci.yml`) runs the **unit tests only** across all projects on every push/PR — no network to any model, no keys, and no Node/CLI required.

### Extra prerequisite for `claude-agent-sdk` only

Live runs need **Node.js 18+ and the Claude Code CLI on PATH**: the Python SDK spawns the CLI as a subprocess. This is inherent to the SDK, not a choice made here — the app code is 100% Python. Unit tests need neither.

## Docs

- `agent-development-guide.md` — the 10 use cases, the original three approaches, scalability, language support. (Predates the Agent SDK addition; see `SPEC.md` §7 for those deltas.)
- `SPEC.md` — project contract, model strategy, testing & cost rules; **§7** covers the `claude-agent-sdk` per-use-case deltas.
- `claude-agent-sdk/_template/README.md` — how this approach is wired: config translation, the `runner` injection seam, and budget discipline.
- `PLAN.md` — phased build order and final status.
- `TRACKING.md` — feasibility matrix & per-cell build status.
- `SECURITY.md` — disclosure policy & scope.
- `docs/security-review.md` — security review vs NIST · OWASP LLM Top 10 · OWASP Web Top 10 (SAST, dependency-CVE, IaC, DAST).

## License & credits

**MIT** — see [`LICENSE`](LICENSE). Copyright (c) 2026 Raghuveer Dendukuri.

**Author:** Raghuveer Dendukuri · **Co-author:** Claude Code (Opus). Every source
file carries an `SPDX-License-Identifier: MIT` header that also names its use case
and links the relevant folder README.
