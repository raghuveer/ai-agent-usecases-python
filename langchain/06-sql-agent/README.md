# UC6 — SQL Agent (langchain)

Natural-language questions over a tiny bundled SQLite store, built the
**langchain** way: an LCEL chain (`prompt | ChatOpenAI | StrOutputParser`)
generates the SQL, then a hand-written safety validator and a read-only SQLite
execution guard run it.

## What it demonstrates

- **Approach trade-off:** langchain makes the generation step compact and
  composable — the prompt/model/parser are one `|`-chain. But the part that
  decides what executes against your database (the single-read-only-SELECT
  validator) is deliberately plain Python, not a framework abstraction, because
  it must be auditable.
- The LLM and the DB connection are **injectable**, so unit tests use
  `FakeListChatModel` + an in-memory fixture DB and run fully offline.

## How it works

1. **Build DB** (`build_db`): execute `data/seed.sql` into an in-memory SQLite
   DB (`customers`, `orders`, ~5 rows each), rebuilt each startup.
2. **Schema inject + generate** (`build_sql_chain`): the LCEL chain inlines the
   introspected schema into the prompt and asks for ONE SELECT.
3. **Validate** (`validate_select`): single read-only SELECT or
   `SQLValidationError` → HTTP 400. Rejects writes/DDL/PRAGMA/multi-statement,
   including comment-smuggled writes.
4. **Execute** (`run_select`): run under a read-only sqlite authorizer.
5. **Summarise** (`summarise`): deterministic NL summary of the rows.

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"06-sql-agent"}`
- `POST /run` body `{"question": str}` →
  `{"sql": str, "rows": list, "explanation": str}`
  (returns **400** when the generated SQL is not a single read-only SELECT)

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (optional) |
| `LLM_MAX_TOKENS` | `256` | max generated tokens (optional) |

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
```

## Feasibility note

Text-to-SQL fits langchain's strength (compose a generation chain) while keeping
the safety guard outside the framework. No Anthropic model is needed; the local
Qwen coder handles this fine.
