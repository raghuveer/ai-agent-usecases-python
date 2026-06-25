# UC6 — SQL Agent (raw-api)

Natural-language questions over a tiny bundled SQLite store, built the
**raw-api** way: the `openai` SDK pointed at the gateway, with schema
introspection, prompt building, SQL safety validation, and read-only execution
all hand-written so you can see exactly what is sent and what runs.

## What it demonstrates

- **Approach trade-off:** raw-api gives you full, explicit control. There is no
  hidden agent loop — `app/sqlagent.py` builds the DB, introspects the schema,
  assembles the prompt, calls `chat.completions.create`, validates the SQL, and
  executes it. The cost is that *you* write every guardrail.
- **Safety is the point:** the generated statement is validated to be exactly
  one read-only `SELECT`. Anything with `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/
  PRAGMA/...` or multiple statements (including comment-smuggled writes) is
  rejected before the DB is touched, and a sqlite authorizer denies writes at
  the engine level as a second line of defence.
- The LLM client and the DB connection are **injectable**, so unit tests run
  fully offline with stubs.

## How it works

1. **Build DB** (`build_db`): execute `data/seed.sql` into an in-memory SQLite DB
   (`customers`, `orders`, ~5 rows each). Rebuilt each startup for determinism.
2. **Schema inject** (`build_prompt`): introspect `sqlite_master` and inline the
   `CREATE TABLE` statements into the prompt.
3. **Generate** (`llm.chat`): one `chat.completions` call asks for ONE SELECT.
4. **Validate** (`validate_select`): single read-only SELECT or `SQLValidationError`
   → HTTP 400.
5. **Execute** (`run_select`): run under a read-only sqlite authorizer.
6. **Summarise** (`summarise`): deterministic NL summary of the rows.

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"06-sql-agent"}`
- `POST /run` body `{"question": str}` →
  `{"sql": str, "rows": list, "explanation": str}`
  (returns **400** when the generated SQL is not a single read-only SELECT)

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


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
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#            -d '{"question":"How many customers are there?"}'
```

## Feasibility note

Text-to-SQL is a good raw-api fit: it is a linear generate → validate → execute
pipeline, and the safety validator is plain code you want to read and audit
directly. No Anthropic model is needed; the local Qwen coder handles this fine.
