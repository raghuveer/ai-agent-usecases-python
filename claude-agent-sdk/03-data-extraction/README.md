# UC03 — data-extraction (claude-agent-sdk) — ⚠️ works, but the harness earns little here

Pull structured invoice data out of raw text, using the Agent SDK's idiom for
structured output: **tool-as-schema**.

## The mechanism

Declare a tool whose `input_schema` *is* the target schema, and tell the agent
its whole job is to call it. The structured record arrives as the tool's
**input** — already a dict:

```
text ──► agent ──► emit_invoice(invoice_number=…, total=…) ──► dict
```

| | raw-api / langchain / langgraph | claude-agent-sdk |
|---|---|---|
| Ask | "reply with JSON matching this schema" | "call this tool" |
| Receive | a string that must be parsed | a dict, via the tool-call protocol |
| Failure modes removed | — | markdown fences, `Here is the JSON:` preambles, trailing prose |
| Failure mode remaining | wrong values | wrong values |

What tool-as-schema removes is *transport* error, not *semantic* error. The agent
can still emit a wrong total, so the tool input is validated against a Pydantic
`Invoice` model and every failure is reported:

- agent never called the tool → `valid: false`, explicit error
- payload violates the schema → `valid: false`, errors with field paths
  (including nested `line_items` entries)
- agent called it twice → the last call wins

`valid: true` is never returned with an unvalidated record.

## Why the caveat

This is an agent harness doing a **one-shot job**. There is no loop to run
(`max_turns` is pinned to 3), and none of the SDK's built-in tools are used. It
works, and the tool-call protocol is a genuinely better transport than
prompt-and-parse — but the harness earns far less here than in the agentic use
cases (02, 07, 08, 10). If extraction is *all* you need, `raw-api/03` is a single
HTTP call and cheaper per document. Reach for this version when extraction is one
step inside a larger agentic flow.

Per this repo's model policy, extraction defaults to a cloud model
(`claude-haiku`): local Qwen was empirically unreliable at strict-schema output.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"03-data-extraction"}`
- `GET /schema` → the tool schema that doubles as the extraction contract
- `POST /run` body `{"document": str}` →
  `{"valid": bool, "invoice": {...}|null, "errors": [str], "num_turns": int,
    "cost_usd": float}`

`data/sample-invoice.txt` is a runnable example document.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | local Qwen is unreliable for strict schemas |
| `AGENT_MAX_TURNS` | `12` | pinned to 3 here — one tool call is the task |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | |

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
#     -d "{\"document\": \"$(cat data/sample-invoice.txt)\"}"
```
