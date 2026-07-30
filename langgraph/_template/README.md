# langgraph `_template` — Phase 0 skeleton

The minimal generic starting point for a **langgraph** use case. Copy this
folder, rename it, and grow the graph. It wires up the shared env contract, an
injectable LLM client, and a one-node `StateGraph` (`echo` → END) behind the
standard HTTP surface.

## What's here

- `app/settings.py` — `pydantic-settings` reading the shared env vars.
- `app/llm.py` — `ChatOpenAI` factory pointed at the gateway; `system_prompt()`
  prepends `/no_think` for `qwen3*` models. Injectable so tests stub it.
- `app/main.py` — FastAPI app + a trivial `StateGraph`:
  - `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"_template"}`
  - `POST /run` `{"question": str}` → `{"answer": "<echo>", "sources": []}`
- `tests/test_unit.py` — mocked via `FakeListChatModel`, no network.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer <key>` |
| `LLM_MODEL` | `qwen3:1.7b` | model id resolved from env |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `512` | max generation tokens |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock
python -m pytest tests/test_unit.py -q        # offline, mocked
uvicorn app.main:app --reload                 # needs a live gateway
```
