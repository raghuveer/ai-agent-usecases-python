# UC9 — recommendations (langgraph)

Content recommendations over a tiny bundled catalog, built with **langgraph**:
the deterministic ranking and the LLM "why" are two nodes of a small
`StateGraph` (`rank` → `explain` → `END`).

## What it demonstrates

- **Determinism where it matters:** the `rank` node is reproducible plain Python
  (`recommend.rank_items`) — overlap between a catalog item's genres/tags and the
  user's liked genres/tags. The model never picks the items.
- **LLM only for the "why":** the `explain` node calls the model once per top-k
  item for a grounded one-sentence reason.
- **Typed graph state** (`RecState`) carries catalog/profile/k in and the
  assembled recommendations out. The LLM and data are **injectable**, so unit
  tests run fully offline with `FakeListChatModel`.
- For RAG-like single-shot work a graph is arguably overkill — it's modelled this
  way for cross-approach comparison with `raw-api/09` and `langchain/09`.

## How it works

1. **Load** (`load_catalog` + `load_profiles`): read `data/catalog.json` (~8
   items) and `data/profiles.json` (~3 users).
2. **`rank` node** (`rank_items`): score each item by overlap with the profile's
   liked genres/tags, drop zero-overlap items, sort (score desc, id asc), top-k.
3. **`explain` node**: one constrained model call per item for a grounded
   one-sentence reason. For qwen3 models we prepend `/no_think`.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"09-recommendations"}`
- `POST /run` body `{"user_id": str, "k": int|null}` (default k=3) →
  `{"recommendations": [{"item_id": str, "title": str, "reason": str}]}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `128` | max generation tokens |
| `REC_TOP_K` | `3` | default number of recommendations |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest -q -m integration

# Serve:
python -m uvicorn app.main:app --reload
```

## Feasibility note

A two-node graph (`rank` → `explain`) is more structure than recommendations
strictly need — the value here is showing the same deterministic-ranking +
LLM-explanation split expressed as a graph. No Anthropic model is needed; local
Qwen writes the reasons fine.
