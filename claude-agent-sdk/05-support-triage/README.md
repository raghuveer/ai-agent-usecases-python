# UC05 — support-triage (claude-agent-sdk)

Classify an inbound support ticket, optionally enrich it from the order system,
and route it — with the routing decision made by the agent rather than by an
`if` statement.

## Agentic routing vs hardcoded routing

The other approaches classify first and then branch in Python:

```python
if category == "shipping" and order_id:
    order = lookup_order(order_id)      # always, by construction
```

Here the agent decides whether an order lookup would actually change its answer,
and only then calls the tool:

```
ticket ──► (maybe) lookup_order ──► emit_triage(category, priority, …)
```

**The trade, stated plainly:** a hardcoded branch is predictable and free; the
agent might skip a lookup you wanted, or make one you did not. So the choice is
made auditable — `order_lookups` in the response reports which orders it actually
fetched. The integration tests assert both directions: a ticket naming a lost
parcel *does* get looked up, a generic shipping question does *not*.

## The decision contract

`emit_triage`'s schema is the contract (same tool-as-schema idea as UC03), and
the result is validated with Pydantic before it leaves the module. `category` and
`priority` are **enums** — an off-list value like `"critical"` is rejected, not
quietly passed through. If the agent never emits a decision, that is reported as
`valid: false` rather than returned as an empty success.

| Field | Values |
|---|---|
| `category` | `billing`, `technical`, `shipping`, `account`, `other` |
| `priority` | `low`, `normal`, `high`, `urgent` |
| `needs_human` | bool — refunds, complaints, legal/DP matters, or uncertainty |
| `reply` | ≤ 3 sentences |

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"05-support-triage"}`
- `GET /schema` → the triage contract
- `GET /orders` → the stand-in order system the lookup tool reads
- `POST /run` body `{"ticket": str}` →
  `{"valid": bool, "decision": {...}|null, "errors": [str],
    "order_lookups": [str], "num_turns": int, "cost_usd": float}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | ample for classification + routing |
| `AGENT_MAX_TURNS` | `12` | ≥5 here: optional lookup, then emit |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | |

Note the model difference from the sibling projects: triage runs on **free local
Qwen** in `raw-api`/`langchain`/`langgraph`, but this approach cannot — the Agent
SDK drives the Claude Code harness, which small local models cannot sustain. See
the root README.

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
#     -d '{"ticket":"Order A-1003 never arrived and I need it by Saturday."}'
```
