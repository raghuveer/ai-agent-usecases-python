# UC3 — data-extraction (langchain)

Extract structured data (a fixed `Invoice` schema) from unstructured invoice
text, built with **langchain**: a small LCEL `prompt | llm | StrOutputParser`
chain produces a JSON string, which is parsed and validated against a Pydantic
`Invoice`, with a single retry that feeds the validation error back to the model.

## What it demonstrates

- **Approach trade-off:** langchain gives you a composable LCEL chain and a clean
  prompt template. The schema is injected as a prompt variable and the chain is
  re-invoked with feedback on the retry. JSON extraction + Pydantic validation
  are still explicit (the local model isn't reliable enough for tool-calling
  structured output, so we parse the text ourselves).
- The chat model is **injectable**, so unit tests use `FakeListChatModel` and run
  fully offline.

## How it works

1. **Chain** (`extract.build_chain`): `ChatPromptTemplate` (system carries the
   `Invoice` JSON schema; for qwen3 models `/no_think` is prepended) → `llm` →
   `StrOutputParser`.
2. **Validate** (`extract.parse_and_validate`): pull the first balanced JSON
   object out of the reply, `json.loads`, `Invoice.model_validate`.
3. **Retry-once** (`extract.extract_invoice`): on any parse/validation failure,
   re-invoke the chain with the error in a `feedback` variable; if it still
   fails, the HTTP layer returns 422.

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

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"03-data-extraction"}`
- `POST /run` body `{"text": str}` → the validated Invoice as JSON plus
  `{"valid": true}`; on parse/validation failure after one retry → `422`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |

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

Extraction maps cleanly onto an LCEL chain, but the local Qwen model is not
reliable for native structured/tool-calling output, so the JSON parsing and
Pydantic validation stay explicit. The retry loop is plain Python around the
chain. No Anthropic model is needed.


## Model note (UC3-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen. Reason:
small local models (`qwen-local-instruct`) are unreliable at emitting strict,
schema-valid JSON for invoice extraction — in testing they echoed malformed
output (e.g. stray `<PERSON>` tokens and collapsed objects), which looked like a
PII guardrail but was actually weak-model behaviour. Haiku extracts the schema
cleanly and ~10x faster. This is the documented "Haiku if JSON flaky" fallback
from `TRACKING.md`. The integration test therefore spends a small, capped amount
of Anthropic budget (marked `anthropic`); unit tests remain fully mocked/offline.
