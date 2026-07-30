# UC07 — multi-agent (claude-agent-sdk) — showcase

A lead agent delegates to specialist subagents and assembles their work into a
report. In this approach the orchestration is **a dict**.

## The team

`ClaudeAgentOptions.agents` takes `{name: AgentDefinition}`. The lead delegates
with the built-in `Agent` tool; the SDK spawns, isolates, and collects each one.

```
lead ──► Agent(researcher) ──► findings ┐
     ──► Agent(analyst)   ──► analysis  ├──► lead writes the report
     ──► Agent(writer)    ──► prose     ┘
```

| Subagent | Tools | Why |
|---|---|---|
| `researcher` | `Grep`, `Glob`, `Read` | reads the corpus; **cannot write** |
| `analyst` | *(none)* | reasoning only, over what it is handed |
| `writer` | *(none)* | prose only, no file access |

## Why this is the native fit

Each subagent gets **its own context window and its own tool allow-list**. That
second part is the real payoff and is genuinely awkward to reproduce elsewhere:
least privilege per role, declared as data. `GET /team` exposes the roster so the
split is inspectable, and a unit test asserts no subagent can `Write` or `Bash`.

| Concern | raw-api | langchain | langgraph | claude-agent-sdk |
|---|---|---|---|---|
| Spawn a specialist | hand-rolled orchestrator (marked impractical) | workaround | sub-graph | `AgentDefinition` entry |
| Isolate its context | manual message-list surgery | partial | separate graph state | automatic per subagent |
| Restrict its tools | manual branching | manual | manual | `tools=[...]` per subagent |
| Invoke it | you write the dispatch | you write the dispatch | edge into the sub-graph | built-in `Agent` tool |

## Corpus

`data/` holds three short markdown notes drawn from this repository's own build
findings. The researcher greps them; nothing reaches the network, so this runs
air-gapped.

## Cost note

This is the most expensive example in the approach — each delegation spawns a
subagent with its own context. `max_turns` is raised to at least 20 here and
`AGENT_MAX_BUDGET_USD` is the backstop. The floor was 12 until a live run hit it:
three delegations plus the lead's own read/write turns intermittently exhausted
the cap, and the run returned an empty report. Turns are the wrong lever for
controlling spend anyway — the budget cap is the one that actually bounds cost.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"07-multi-agent"}`
- `GET /team` → the roster: each subagent's description and tool allow-list
- `POST /run` body `{"question": str}` →
  `{"report": str, "subagents_used": [str], "tools_used": [str],
    "num_turns": int, "cost_usd": float}`

`subagents_used` is read back out of the `Agent` tool calls, so it reflects what
the lead *actually* delegated, not what it was told to do.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | `claude-sonnet` if delegation quality is poor |
| `AGENT_MAX_TURNS` | `12` | raised to ≥20 here for fan-out |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | raise for harder questions |

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent team (gateway + key + Node/CLI; the priciest example here):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl localhost:8000/team
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"question":"What did we learn about local models and ReAct loops?"}'
```
