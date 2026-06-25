# UC6 — SQL Agent (langgraph)

Natural-language questions over a tiny bundled SQLite store, built as a tiny
**langgraph** `StateGraph`. The agent's control flow is modelled as explicit
edges:

```
generate -> validate --(ok)--> execute -> END
                     \--(bad)--> END   (state carries the rejection)
```

## What it demonstrates

- **Approach trade-off:** langgraph makes the "validate before you execute"
  branch a first-class, inspectable conditional edge rather than a buried `if`.
  That is the natural framing for an agent that must gate a side effect (running
  SQL) on a safety check.
- The load-bearing safety check (single read-only SELECT) is still plain,
  auditable Python inside the `validate` node — the graph just routes on its
  result.
- The LLM and DB connection are **injected** into `create_app`, so unit tests
  use `FakeListChatModel` + an in-memory fixture DB and run fully offline.

## How it works

1. **Build DB** (`build_db`): execute `data/seed.sql` into an in-memory SQLite
   DB (`customers`, `orders`, ~5 rows each), rebuilt each startup.
2. **generate** node: inline the introspected schema into the prompt, ask the
   LLM for ONE SELECT, extract it from any ```sql fence.
3. **validate** node: `validate_select` accepts a single read-only SELECT or
   sets `error`; the conditional edge routes to `execute` or to `END`. Rejects
   writes/DDL/PRAGMA/multi-statement, including comment-smuggled writes.
4. **execute** node: run under a read-only sqlite authorizer, summarise the rows.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"06-sql-agent"}`
- `POST /run` body `{"question": str}` →
  `{"sql": str, "rows": list, "explanation": str}`
  (returns **400** when the generated SQL is not a single read-only SELECT)

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
python -m uvicorn app.main:app --reload
```

## Feasibility note

Text-to-SQL maps cleanly onto a graph because the safety gate is a real branch in
the control flow. No Anthropic model is needed; the local Qwen coder handles this
fine.
