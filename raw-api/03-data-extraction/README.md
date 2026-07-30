# UC3 — data-extraction (raw-api)

Extract structured data (a fixed `Invoice` schema) from unstructured invoice
text, built the **raw-api** way: the `openai` SDK pointed at the gateway, with
prompt building, JSON extraction, Pydantic validation, and the retry loop all
hand-written so you can see exactly what is sent and how output is validated.

## What it demonstrates

- **Approach trade-off:** raw-api gives you full, explicit control. There is no
  hidden chain — `app/extract.py` hands the model the JSON schema, pulls the
  first balanced `{...}` object out of the reply by hand, validates it against a
  fixed Pydantic `Invoice`, and retries the model once with the validation error
  appended. The cost is that *you* write every step.
- The LLM call is **injectable**, so unit tests run fully offline with stubs.

## How it works

1. **Prompt** (`extract.SYSTEM_PROMPT`): the model is given the `Invoice` JSON
   schema and told to return ONLY a JSON object.
2. **Extract** (`extract.extract_json_object`): tolerate markdown fences / prose
   and pull the first balanced JSON object.
3. **Validate** (`extract.parse_and_validate`): `json.loads` + `Invoice.model_validate`.
4. **Retry-once** (`extract.extract_invoice`): on any parse/validation failure,
   call the model ONE more time with the error appended; if it still fails, the
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

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"03-data-extraction"}`
- `POST /run` body `{"text": str}` → the validated Invoice as JSON plus
  `{"valid": true}`; on parse/validation failure after one retry → `422`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `LLM_STRUCTURED_MODE` | `text` | `text` (parse JSON from reply) or `native` (provider JSON mode) |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Structured-output modes

`LLM_STRUCTURED_MODE` selects how the JSON object is obtained. Both modes use the
same `Invoice` schema and the same validate + retry-once contract; only the
parsing path differs.

- **`text`** (default): the system prompt instructs the model to return JSON, and
  the code pulls the JSON object out of the reply (tolerating prose/markdown
  fences). Portable across **any** chat model — this is what the integration test
  exercises.
- **`native`**: passes `response_format={"type": "json_object"}` (OpenAI-compatible
  JSON mode) so the provider guarantees a JSON object, which is then `json.loads`-d
  directly. More reliable, but requires provider support — OpenAI JSON mode, many
  LiteLLM routes, and Anthropic via the gateway support it; **not all local models
  do**. Switch with `LLM_STRUCTURED_MODE=native` in `.env`; the default stays
  `text`.


## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#            -d "{\"text\":\"INVOICE #INV-1 Total due: 10.00 ...\"}"
```

## Feasibility note

Structured extraction is a good fit for raw-api: it is a linear prompt → parse →
validate → (maybe) retry pipeline. The only real complexity is robust JSON
extraction and validation, which is plain Python here. No Anthropic model is
needed; the local Qwen instruct model handles JSON extraction fine.


## Model note (UC3-specific)

This use case defaults to **`claude-haiku`**, not the free local Qwen. Reason:
small local models (`qwen-local-instruct`) are unreliable at emitting strict,
schema-valid JSON for invoice extraction — in testing they echoed malformed
output (e.g. stray `<PERSON>` tokens and collapsed objects), which looked like a
PII guardrail but was actually weak-model behaviour. Haiku extracts the schema
cleanly and ~10x faster. This is the documented "Haiku if JSON flaky" fallback
from `TRACKING.md`. The integration test therefore spends a small, capped amount
of Anthropic budget (marked `anthropic`); unit tests remain fully mocked/offline.
