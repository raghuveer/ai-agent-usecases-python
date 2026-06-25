# raw-api `_template` — Phase 0 skeleton

The minimal generic starting point for a **raw-api** use case. Copy this folder,
rename it (`raw-api/NN-something/`), and grow `app/` into the real logic.

## What it gives you

- `app/settings.py` — `pydantic-settings` reading the shared env contract.
- `app/llm.py` — an **injectable** `openai` SDK client pointed at the gateway,
  plus a `chat()` helper and qwen3 `/no_think` handling. The client lives on
  `app.state.client`, so unit tests swap in a stub and never touch the network.
- `app/main.py` — FastAPI app with `GET /health` and `POST /run`. `/run` just
  echoes the question through the (mockable) LLM and returns
  `{"answer": ..., "sources": []}`.
- `tests/test_unit.py` — mocked, offline, passing.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen3:1.7b` | model id (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `RAG_TOP_K` | `3` | unused here; kept for the shared contract |
| `CHROMA_DIR` | `.chroma` | unused here; kept for the shared contract |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


## Run

```bash
python -m uv venv
python -m uv pip install -e ".[dev]"
python -m pytest tests/test_unit.py -q          # offline, must pass
python -m uvicorn app.main:app --reload         # needs a live gateway
```
