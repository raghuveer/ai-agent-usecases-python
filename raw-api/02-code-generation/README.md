# UC2 — code-generation (raw-api)

Generate code for a natural-language task, built the **raw-api** way: the
`openai` SDK pointed at the gateway, with the prompt, fenced-block extraction,
and the optional safety check all hand-written so you can see exactly what runs.

## What it demonstrates

- **Approach trade-off:** raw-api gives you full, explicit control. There is no
  hidden chain — `app/codegen.py` builds the prompt string, calls
  `chat.completions.create`, and pulls the code out of the first fenced block
  with a regex. The cost is that *you* write every step.
- The LLM client is **injectable**, so unit tests run fully offline with a stub.
- **Safety:** by default `/run` does NOT execute generated code. An OPTIONAL
  self-check (`RUN_CODE_CHECK=1`, python only) smoke-runs the code in a
  subprocess with a timeout and iterates on failure (capped at 2 attempts).

## How it works

1. **Prompt** (`codegen.build_prompt`): instruct the coder model to return one
   fenced code block plus a short explanation.
2. **Generate** (`llm.chat`): one `chat.completions` call. For qwen3 models we
   prepend `/no_think` to the system prompt.
3. **Extract** (`codegen.extract_code` / `extract_explanation`): take the first
   ```` ``` ```` block as the code; treat trailing prose as the explanation.
4. **(Optional) Check** (`codegen.smoke_check_python`): when enabled, run the
   code once in a subprocess; on failure ask the model to fix it (max 2 tries).

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"02-code-generation"}`
- `POST /run` body `{"task": str, "language": str|null}` (default language
  `python`) → `{"code": str, "language": str, "explanation": str, "tests_passed": bool|null}`

`tests_passed` is `null` unless `RUN_CODE_CHECK=1` and the language is python.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `DEFAULT_LANGUAGE` | `python` | language used when the request omits one |
| `RUN_CODE_CHECK` | `0` | `1` smoke-runs generated python (subprocess) |
| `CODE_CHECK_TIMEOUT` | `10` | seconds before the smoke run is killed |

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
#            -d '{"task":"Write a Python function add(a,b) that returns their sum"}'
```

## Feasibility note

Code generation is a clean fit for raw-api: it's a linear prompt → generate →
extract pipeline. The only branching is the optional self-check loop, which stays
readable hand-written. No Anthropic model is needed; the free local Qwen coder
handles the task fine.
