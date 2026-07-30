# UC09 — recommendations (claude-agent-sdk) — ⚠️ works, modest win over a single call

Personalised, explained product recommendations. The agent fetches the user's
profile and the catalog through tools, ranks, and justifies each pick.

```
get_profile ──► list_catalog ──► emit_recommendations(items, rationale)
```

## Grounding is the point

Neither the profile nor the catalog appears in the prompt — the agent has to
fetch both. That keeps the explanations tied to fetched data, and it makes the
output checkable: **every recommended id is validated against the real catalog**
before the response is built. An invented product id fails the whole response
(`valid: false`) rather than being enriched and returned; a partially
hallucinated list is rejected wholesale.

Recommended items are then enriched from the catalog — name, category, price come
from *our* data, never from the model's claim about it.

## Why the caveat

Ranking a seven-item catalog does not need an agent loop. A single prompt with the
catalog inlined would be cheaper and faster. What the loop actually buys here:

- explanations grounded in fetched data rather than prompt text
- the agent consulting only the categories the profile implies, instead of
  shipping the whole catalog on every request

On a catalog this size that is a modest win; on a large or permissioned catalog
it becomes a real one. If you only need ranking, `raw-api/09` is one HTTP call.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"09-recommendations"}`
- `GET /catalog` → the seven products
- `GET /profiles` → the stand-in profile store
- `POST /run` body `{"user_id": str}` →
  `{"valid": bool, "items": [{"id","name","category","price","reason"}],
    "rationale": str, "errors": [str], "num_turns": int, "cost_usd": float}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | ample for ranking + explanation |
| `AGENT_MAX_TURNS` | `12` | profile, catalog, emit |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | |

As with UC05, the sibling projects run this on free local Qwen; this approach
cannot — see the root README.

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent (gateway + key + Node/CLI; spends a little budget):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"user_id":"u-1"}'
```
