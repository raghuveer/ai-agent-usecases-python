# UC3 — data-extraction (langgraph)

Extract structured data (a fixed `Invoice` schema) from unstructured invoice
text, built as a tiny **langgraph** `StateGraph`. The validate-and-retry loop is
the graph topology, not an imperative try/except:

```
extract -> validate -> (invalid & attempts left? back to extract : END)
```

## What it demonstrates

- **Approach trade-off:** langgraph makes the retry-on-invalid control flow
  explicit as edges. `extract` calls the LLM (with the schema + any prior
  validation error), `validate` parses/validates against the Pydantic `Invoice`,
  and a conditional edge loops back to `extract` exactly once. This is overkill
  for one retry, but shows how the same loop scales to multi-step validation.
- The chat model is **injectable**, so unit tests use `FakeListChatModel` and run
  fully offline.

## How it works

1. **extract node**: build messages (system carries the `Invoice` JSON schema;
   `/no_think` is prepended for qwen3 models) and call `llm.invoke`; bump
   `attempts`.
2. **validate node** (`extract.parse_and_validate`): pull the first balanced JSON
   object out of the reply, `json.loads`, `Invoice.model_validate`; store the
   invoice or the error.
3. **conditional edge** (`route`): valid → END; invalid with attempts left
   (`MAX_ATTEMPTS = 2`) → back to `extract`; otherwise END. If still invalid the
   HTTP layer returns 422.

## Schema

```
Invoice = {
  invoice_number: str,
  vendor: str,
  date: str,
  total: float,
  line_items: list[{description: str, amount: float}],
}
```

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"03-data-extraction"}`
- `POST /run` body `{"text": str}` → the validated Invoice as JSON plus
  `{"valid": true}`; on parse/validation failure after one retry → `422`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `512` | max generation tokens |
| `LLM_STRUCTURED_MODE` | `text` | `text` \| `native` (see below) |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL`
(and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes.
See the root README's "Use a different model or provider" table.

## Structured-output modes

`LLM_STRUCTURED_MODE` selects how the `Invoice` schema is extracted:

- **`text`** (default): the prompt instructs the model to return a single JSON
  object and the reply is parsed from text. Portable across **any** chat model;
  reproduces this project's original behavior.
- **`native`**: uses `llm.with_structured_output(Invoice)` (langchain's idiomatic
  native structured output), which returns a validated object directly. More
  reliable, but requires provider support (OpenAI JSON mode, many LiteLLM
  routes; not all local models). The same `Invoice` schema and validate/
  retry-once contract apply in both modes.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

## Feasibility note

Extraction with validate-and-retry is a genuine fit for a graph: the retry loop
is a cycle, which is exactly what langgraph models cleanly. For a single retry it
is more machinery than needed, but it generalises to richer correction loops. The
local Qwen instruct model handles JSON extraction fine; no Anthropic model is
needed.


## Model note (UC3-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen. Reason:
small local models (`qwen-local-instruct`) are unreliable at emitting strict,
schema-valid JSON for invoice extraction — in testing they echoed malformed
output (e.g. stray `<PERSON>` tokens and collapsed objects), which looked like a
PII guardrail but was actually weak-model behaviour. Haiku extracts the schema
cleanly and ~10x faster. This is the documented "Haiku if JSON flaky" fallback
from `TRACKING.md`. The integration test therefore spends a small, capped amount
of Anthropic budget (marked `anthropic`); unit tests remain fully mocked/offline.
