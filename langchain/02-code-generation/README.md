# UC2 — code-generation (langchain)

Generate code for a natural-language task, built with **langchain**: a plain
LCEL chain `ChatPromptTemplate | llm | StrOutputParser` against the gateway, with
fenced-block extraction and an optional safety check around it.

## What it demonstrates

- **Approach trade-off:** langchain gives you a composable chain and a uniform
  `ChatOpenAI` client. The prompt/parse plumbing is declarative LCEL
  (`app/codegen.build_chain`); you still hand-write the fenced-block extraction
  and the optional self-check.
- The LLM is **injectable**, so unit tests swap in `FakeListChatModel` and run
  fully offline.
- **Safety:** by default `/run` does NOT execute generated code. An OPTIONAL
  self-check (`RUN_CODE_CHECK=1`, python only) smoke-runs the code in a
  subprocess with a timeout and iterates on failure (capped at 2 attempts).

## How it works

1. **Chain** (`codegen.build_chain`): `ChatPromptTemplate | llm | StrOutputParser`.
   For qwen3 models the system message is prefixed with `/no_think`.
2. **Generate**: `chain.invoke({"task", "language"})` returns the raw text.
3. **Extract** (`codegen.extract_code` / `extract_explanation`): take the first
   ```` ``` ```` block as the code; treat trailing prose as the explanation.
4. **(Optional) Check** (`codegen.smoke_check_python`): when enabled, run the
   code once in a subprocess; on failure re-run the chain (max 2 tries).

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"02-code-generation"}`
- `POST /run` body `{"task": str, "language": str|null}` (default language
  `python`) → `{"code": str, "language": str, "explanation": str, "tests_passed": bool|null}`

`tests_passed` is `null` unless `RUN_CODE_CHECK=1` and the language is python.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-coder` | model alias (free local Qwen coder) |
| `DEFAULT_LANGUAGE` | `python` | language used when the request omits one |
| `RUN_CODE_CHECK` | `0` | `1` smoke-runs generated python (subprocess) |
| `CODE_CHECK_TIMEOUT` | `10` | seconds before the smoke run is killed |

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

Code generation maps cleanly to a single LCEL chain; the framework value here is
modest (one prompt → one model → one parser). The only branching is the optional
self-check loop, written in plain Python around the chain. No Anthropic model is
needed; the free local Qwen coder handles the task fine.
