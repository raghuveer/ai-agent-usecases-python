# langchain `_template` — Phase 0 skeleton

The minimal generic starting point for every langchain use case in this repo.
Copy this folder, rename it, and grow the `app/` logic for your use case.

## What it gives you

- `app/settings.py` — `pydantic-settings` config reading the shared env contract.
- `app/llm.py` — an **injectable** `ChatOpenAI` factory pointed at the gateway.
- `app/main.py` — FastAPI app with `GET /health` and `POST /run` (echoes the
  question back through the mockable LLM; no retrieval yet).
- `tests/test_unit.py` — mocked LLM (`FakeListChatModel`), no network, passes offline.

## Env vars

See `.env.example`:

| var | meaning |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible gateway base (`.../v1`) |
| `LLM_GATEWAY_KEY` | bearer key (placeholder is fine for local/unit) |
| `LLM_MODEL` | model id; `qwen3:*` triggers `/no_think` system-prompt handling |
| `RAG_TOP_K` | retrieval depth (carried for use cases that copy this) |
| `CHROMA_DIR` | vector store dir (carried for use cases that copy this) |
| `LLM_TEMPERATURE` | sampling temperature (optional, default `0.0`) |
| `LLM_MAX_TOKENS` | max generated tokens (optional, default `256`) |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv venv
python -m uv pip install -e .
python -m pytest tests/test_unit.py -q          # offline, mocked

# live (needs the local gateway running):
uvicorn app.main:app --reload
```

## Design notes

- The LLM client is built in the FastAPI lifespan and stored on `app.state.llm`,
  so unit tests inject a `FakeListChatModel` before the first request — keeping
  tests deterministic and network-free.
- qwen3 "thinking" mode is disabled by prepending `/no_think` to the system
  prompt when `LLM_MODEL` starts with `qwen3`.
