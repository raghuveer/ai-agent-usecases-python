# UC2 — code-generation (langgraph)

Generate code for a natural-language task, built as a **langgraph** `StateGraph`
with a real `generate -> check -> (loop | END)` cycle.

## What it demonstrates

- **Approach trade-off:** this is where langgraph earns its keep. The OPTIONAL
  self-check / fix loop is a genuine cyclic graph with a conditional edge, not a
  hand-rolled `for` loop. `generate` produces code, `check` runs the smoke test
  (when enabled), and a conditional edge routes back to `generate` on failure
  until the cap is hit.
- The LLM and the code executor are both **injected**, so unit tests swap in
  fakes and run fully offline and never execute generated code.
- **Safety:** by default `/run` does NOT execute generated code — the `check`
  node is a pass-through (`tests_passed=null`). Enable with `RUN_CODE_CHECK=1`
  (python only), capped at 2 iterations.

## How it works

```
generate --> check --(fail & under cap)--> generate
                   \--(pass | flag off | cap)--> END
```

1. **generate** (`codegen.build_codegen_graph`): one `llm.invoke` with the coder
   system prompt (qwen3 → `/no_think`); extract the first fenced block + prose.
2. **check**: when `run_code_check` and language==python, smoke-run the code in a
   subprocess; otherwise return `tests_passed=null`.
3. **conditional edge**: on a failing check under the iteration cap, loop back to
   `generate` with the failing code appended so the model fixes it.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"02-code-generation"}`
- `POST /run` body `{"task": str, "language": str|null}` (default language
  `python`) → `{"code": str, "language": str, "explanation": str, "tests_passed": bool|null}`

`tests_passed` is `null` unless `RUN_CODE_CHECK=1` and the language is python.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | max generation tokens |
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
```

## Feasibility note

Code generation with an iterate-on-failure loop is the use case that most
justifies langgraph: the cycle and the conditional edge express the retry logic
declaratively. Without the self-check loop the graph degenerates to two linear
nodes. No Anthropic model is needed; the free local Qwen coder handles the task.
