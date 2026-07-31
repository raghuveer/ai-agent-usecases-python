# UC06 — sql-agent (claude-agent-sdk)

Natural-language questions answered against a bundled SQLite database, using
three custom SDK tools.

## Schema discovery is the agent's job

```
list_tables ──► describe_table ──► run_select ──► answer
```

The other approaches typically inject the whole schema into the system prompt on
every request, whether the question needs it or not. Here the agent looks up only
what it needs. That scales better to wide schemas and costs a turn or two more on
narrow ones — a real trade, not a free win.

## Two independent read-only defences

The SQL is written by a model, so one guard is not enough:

1. **Syntactic** (`assert_read_only`) — must be a single `SELECT` / `WITH`
   statement. Stacked statements (`SELECT 1; DROP TABLE …`) are rejected, and
   non-read keywords are matched as whole words so a column named `ordered_on`
   is not mistaken for `UPDATE`.
2. **Driver-level** (`connect_read_only`) — SQLite is opened with `mode=ro`, so a
   write that somehow passed (1) still fails at the driver.

(1) alone would be a blocklist, which is the weaker pattern; (2) is what actually
holds. Both are unit-tested, including `DROP`/`DELETE`/`ATTACH`/`PRAGMA`, stacked
statements, and a table-name injection attempt against `describe_table`.

A rejected query returns an **error tool result**, not an exception — so the
agent reads why and rewrites the query instead of the request failing.

## Database

Generated from `SEED_SQL` into `data/shop.db` on first use (not committed, so
clones stay reproducible and no binary lands in git). Three tables — `customers`,
`products`, `orders` — with a handful of rows. `GET /schema` shows it.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"06-sql-agent"}`
- `GET /schema` → tables and columns
- `POST /run` body `{"question": str}` →
  `{"answer": str, "queries": [str], "tools_used": [str], "num_turns": int,
    "cost_usd": float, "stop_reason": str}`

`queries` is the SQL the agent actually executed — the audit trail for its answer.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | `claude-sonnet` for gnarlier joins |
| `AGENT_MAX_TURNS` | `12` | raised to ≥8: discover, query, and possibly fix |
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
#   curl localhost:8000/schema
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"question":"Which country has the most customers?"}'
```
