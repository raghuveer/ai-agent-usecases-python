# UC08 — autonomous-react (claude-agent-sdk) — showcase

An agent that reasons, calls tools, observes results, and repeats until it can
answer — **with no loop in this repository**.

## Why this is the native fit

The Agent SDK *is* a ReAct loop. `app/react_agent.py` registers two tools and a
prompt; the model decides what to call and when to stop, and `max_turns` bounds
it. What that removes is not cosmetic — the text-ReAct implementations in this
repo needed every one of the following, and none of it exists here:

| Text-ReAct (raw-api / langchain / langgraph) | claude-agent-sdk |
|---|---|
| Parser for `Thought:` / `Action:` / `Arguments:` lines | — tool calls are structured protocol messages |
| `stop=["Observation:"]`, or the model invents its own tool results | — the runtime supplies real results |
| Renaming `Action Input` → `Arguments`, because the gateway's PII redaction masked that literal string to `<PERSON>` and broke parsing | — no prose field names to mangle |
| Hand-written step ceiling + "never reached Final Answer" fallback | `max_turns`, surfaced as `hit_turn_limit` |

That third row is a real defect this repo hit, recorded in its own build notes:
prompt-formatted ReAct puts the control protocol *in the text*, where anything
that rewrites text can break it. Structured tool calls sidestep the whole class.

## Tools

| Tool | Purpose |
|---|---|
| `lookup_metric(name)` | fetch one metric from a fixed warehouse (`GET /metrics`) |
| `calculate(expression)` | arithmetic only |

The agent cannot see metric values in its prompt, so answering requires real tool
calls — which is what the integration test asserts.

**`calculate` does not use `eval()`.** It parses to an AST and permits only
numeric literals and `+ - * / ** unary-minus`; anything else raises. It evaluates
model-authored text, so a unit test throws import tricks, attribute walks, and
`__subclasses__()` at it. Tool errors come back as `is_error` results rather than
exceptions, so the agent can correct itself instead of the run dying.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"08-autonomous-react"}`
- `GET /metrics` → the fixed warehouse the agent must query through tools
- `POST /run` body `{"question": str}` →
  `{"answer": str, "trace": [{"tool": str, "input": {...}}], "num_turns": int,
    "cost_usd": float, "hit_turn_limit": bool}`

`trace` is the observed sequence of tool calls — the ReAct trajectory, read back
from what the agent actually did. `hit_turn_limit` is reported rather than
hidden: a truncated run should be visible, not silently returned as an answer.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | `claude-sonnet` for longer chains |
| `AGENT_MAX_TURNS` | `6` | raised to ≥8 here; a multi-step chain needs the room |
| `AGENT_MAX_BUDGET_USD` | `0.25` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | raise for harder questions |

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent (gateway + key + Node/CLI; spends a little budget):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"question":"What is our gross margin as a percentage of revenue?"}'
```
