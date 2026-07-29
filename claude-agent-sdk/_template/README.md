# claude-agent-sdk — project template

The skeleton every `claude-agent-sdk/<usecase>/` folder is copied from. It is a
one-shot agent with no tools: `POST /run` sends the question and returns the
answer. Copy this folder to start a new use case.

## What is different about this approach

The other three approaches send messages to a model and own the control flow. The
Agent SDK **is** the control flow: you configure an agent (system prompt, tools,
budget, permissions) and it runs its own loop.

Concretely, this template establishes three things every use case reuses.

### 1. Config translation (`app/agent.py` → `sdk_env`)

The SDK does not take a base URL or a key as Python arguments. It spawns the
Claude Code CLI, and *that process* reads the environment. So the repo-standard
`LLM_*` settings are translated into the variables the subprocess reads:

| Repo setting | SDK env var |
|---|---|
| `LLM_BASE_URL` | `ANTHROPIC_BASE_URL` |
| `LLM_GATEWAY_KEY` | `ANTHROPIC_AUTH_TOKEN` |

Two gotchas worth internalising:

- **`ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.** The gateway requires
  `Authorization: Bearer`. `ANTHROPIC_API_KEY` makes the CLI send `x-api-key`,
  which the gateway rejects with 401.
- **`LLM_BASE_URL` has no `/v1` suffix here.** This is the Anthropic surface and
  the SDK appends `/v1/messages` itself. The other three approaches use the
  OpenAI surface and *do* include `/v1`.

### 2. The injection seam (`Runner`)

`query()` spawns a subprocess, which unit tests must never do. So every
`create_app()` takes a `runner` callable:

```python
Runner = Callable[[str, ClaudeAgentOptions], Awaitable[AgentResult]]
```

Production uses `default_runner` (`query()` → `collect()`). Unit tests inject a
stub. This mirrors the injectable LLM client in the other three approaches: the
tests are fully offline — no CLI, no network, no key.

`collect()` is kept separate and **pure over public SDK types**, so tests
exercise it with real `AssistantMessage` / `ResultMessage` objects and still
verify the parsing against what the CLI actually emits.

### 3. Budget discipline (`build_options`)

An agent loop makes an unbounded number of model calls, so both a turn cap and a
hard dollar cap are set on every run. `build_options` also pins:

- `setting_sources=[]` — stops the SDK reading `settings.json` from the
  developer's `~/.claude` and the repo's `.claude/`.
- `CLAUDE_CONFIG_DIR` pointed at a throwaway directory (in `sdk_env`) — **this is
  what actually isolates the run**, and it is easy to get wrong.

> ⚠️ **`setting_sources=[]` is not enough on its own.** It gates *settings files*
> only. It does **not** stop the CLI loading the developer's `~/.claude` project
> **memory** or a parent `CLAUDE.md`. Verified empirically: with
> `setting_sources=[]` set, a probe agent recited this repository's private
> memory index verbatim. Pointing `CLAUDE_CONFIG_DIR` at an empty directory
> returns `NONE VISIBLE`.
>
> This matters twice over — **reproducibility** (a run must not depend on whose
> laptop it is on; UC07 was answering from leaked memory instead of its own
> corpus) and **disclosure** (developer memory could otherwise be echoed into an
> API response). See `docs/security-review.md` F14.
- `tools=[]` — start from no built-in tools; each use case opts in explicitly, so
  the README's tool list is the truth.

## Why local Qwen is not an option here

The other three approaches default to free local Qwen. This one cannot. The SDK
drives the Claude Code harness — a large system prompt plus a built-in tool loop
— and small local models cannot sustain multi-step tool use (this repo's own
finding, recorded in `TRACKING.md`). So this approach defaults to `claude-haiku`
and every live test spends a small, capped amount of budget.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"_template"}`
- `POST /run` body `{"question": str}` →
  `{"answer": str, "tools_used": [str], "num_turns": int, "cost_usd": float}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | allow-listed alias |
| `AGENT_MAX_TURNS` | `6` | hard cap on agent turns |
| `AGENT_MAX_BUDGET_USD` | `0.25` | hard cap on spend per run |
| `AGENT_EFFORT` | `low` | thinking depth (`low`…`max`) |

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs. The Python SDK
  spawns the CLI as a subprocess; this is inherent to the SDK. Unit tests need
  neither.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

## Troubleshooting live runs

**`API Error: 401 invalid virtual key`** — the gateway reached you, so the URL and
transport are fine; the key is wrong or expired. Mint a fresh `sk-aiup-…` with
`scripts/seed-virtual-keys.mjs` in the platform repo. If you instead see
`missing_virtual_key`, the CLI sent `x-api-key` — you set `ANTHROPIC_API_KEY`
somewhere; unset it and use `ANTHROPIC_AUTH_TOKEN`.

**A bad key makes the call hang for minutes.** The CLI retries auth failures with
backoff. While debugging, add to `.env` (or `ClaudeAgentOptions.env`):

```
CLAUDE_CODE_MAX_RETRIES=0
API_TIMEOUT_MS=8000
```

That turns a multi-minute hang into a sub-second error. Don't ship these in
production config — retries are useful once the key is right.

**Auth failures raise, they do not return.** `query()` raises rather than
yielding a `ResultMessage` with `is_error=True`, so a misconfigured deployment
surfaces as a 500 from `/run` rather than a well-formed error body. That is
deliberate here: misconfiguration should fail loudly, not look like a bad answer.
