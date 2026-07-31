# UC02 — code-generation (claude-agent-sdk) — showcase

Generate code, generate tests for it, **run them**, and fix what fails — with no
loop code in this repository.

## Why this is the native fit

In the other three approaches, "generate code and check it" means you build the
harness: a subprocess sandbox, a timeout, a test-output parser, and a bounded
retry loop. The Agent SDK ships all of that. Give the agent a working directory
and a definition of done, and it drives itself:

```
write solution.py ──► write test_solution.py ──► Bash: pytest
         ▲                                          │
         └───────────── fix on failure ◄────────────┘
```

`app/codegen.py` contains **no loop**. `max_turns` bounds the iteration; the
built-in `Write` / `Read` / `Edit` / `Bash` tools do the work.

| Concern | raw-api / langchain / langgraph | claude-agent-sdk |
|---|---|---|
| Execute generated code | hand-written `subprocess` + timeout | built-in `Bash` tool |
| Write files | parse code out of the reply, write it yourself | built-in `Write` / `Edit` |
| Iterate on failures | hand-written retry loop with an iteration cap | the agent loop, bounded by `max_turns` |
| Know it actually passed | parse pytest stdout | agent runs it; we verify artefacts + that `Bash` ran |

## Verifying success honestly

`tests_passed` is deliberately conservative. It requires **all** of: both files
exist, the agent actually invoked `Bash` (so it ran the tests rather than merely
claiming they would pass), the run did not error, and it did not stop on
`max_turns`. An agent that says "the tests pass" without running them does not
satisfy this.

## `cwd` is not a sandbox — verified the hard way

Each run gets a fresh `tempfile.mkdtemp()` as `cwd`, and the tool allow-list omits
`WebFetch`/`WebSearch`. **That does not confine the agent.** On a live run the
`Write` tool accepted an *absolute* path and the model wrote to
`/tmp/solution.py` instead of the workdir — sometimes self-correcting via Bash,
sometimes not, which made results flaky. Two mitigations:

1. The system prompt demands bare relative filenames and forbids absolute paths
   and `/tmp`.
2. Artefacts are only read back from the workdir, so anything written outside it
   does not count as output and `tests_passed` stays `false`.

Neither is a security boundary. So the SDK's own `sandbox` is applied too, and
it is **on by default** here.

## The shell is sandboxed by default (F9)

This is the only project in the repo that grants `Bash`, and it grants it to a
loop driven by request text you did not write. Prompt injection in `POST /run`
`{"task": ...}` is therefore a *command execution* risk, not a misinformation
one — so the sandbox is not something a deployer has to remember to switch on:

| setting | value | why |
|---|---|---|
| `enabled` | `true` | on by default; `SANDBOX_BASH=false` to opt out |
| `autoAllowBashIfSandboxed` | `true` | the agent is *meant* to run pytest; prompting per command breaks the loop. Only acceptable because what is auto-approved is a *sandboxed* command |
| `allowUnsandboxedCommands` | **`false`** | the SDK default is `true`, which lets any command opt out via `dangerouslyDisableSandbox`. The thing choosing commands is model output — a control the attacker can ask to have disabled is not a control |
| `network.allowedDomains` | `[]` | writing and testing a function needs no network; denying it removes the exfiltration half of an injection payload |

**Every response reports `sandboxed`, and it is observed, not assumed.** The SDK
sandboxes bash on **macOS/Linux only**. Forcing the request through on Windows,
the CLI answered:

```
⚠ Sandbox disabled: sandbox is enabled but the Windows sandbox is not active
  on this session (feature gate off)
  Commands will run WITHOUT sandboxing.
```

…and the first version of this code still reported `sandboxed: true`, because it
was describing its own config. The CLI announces the downgrade on stderr and the
SDK forwards stderr to a callback, so the real answer is available for the
asking — `sandbox_note` carries the CLI's reason verbatim. Claiming a boundary
that is not there is worse than having none: it is the claim a deployer acts on.

**When `sandboxed` is false, nothing has changed from the paragraph above** — run
it in a disposable, network-isolated, resource-capped container or VM.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"02-code-generation"}`
- `POST /run` body `{"task": str}` →
  `{"solution": str, "tests": str, "summary": str, "tests_passed": bool,
    "files": [str], "tools_used": [str], "num_turns": int, "cost_usd": float,
    "stop_reason": str, "sandboxed": bool, "sandbox_note": str|null}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | `claude-sonnet` for harder tasks |
| `AGENT_MAX_TURNS` | `12` | **set this to ~25 here** — see below |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | raise for harder tasks |
| `SANDBOX_BASH` | `true` | confine the shell — see above; macOS/Linux only |
| `SANDBOX_ALLOW_NETWORK` | `false` | the task needs no network |

**Measured on a live run**, this is the most turn-hungry use case in the approach:
a FizzBuzz task took **9–10 turns and $0.35–0.48** on `claude-haiku`, because the
agent writes, runs, finds its own test bug, fixes, and re-runs. At the shared
default of 12 turns it can exhaust the cap mid-fix, so `.env` here uses
`AGENT_MAX_TURNS=25`. Cost is inflated by the local gateway not passing prompt
caching through (every turn re-pays the harness prompt); going direct to
`api.anthropic.com` is materially cheaper.

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent (gateway + key + Node/CLI; spends a little budget):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"task":"Write a function fizzbuzz(n) returning a list of strings."}'
```
