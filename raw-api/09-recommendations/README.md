# UC9 — recommendations (raw-api)

Content recommendations over a tiny bundled catalog, built the **raw-api** way:
the deterministic ranking is plain Python, and the `openai` SDK (pointed at the
gateway) is used ONLY to write a one-sentence reason per recommended item.

## What it demonstrates

- **Determinism where it matters:** scoring/ranking is reproducible plain Python
  (`recommend.rank_items`) — overlap between the catalog item's genres/tags and
  the user's liked genres/tags. The model never picks the items, so results are
  stable and explainable.
- **LLM only for the "why":** for each top-k item, `recommend.reason_for` asks the
  model for one grounded sentence, constrained to the supplied profile/item data.
- The LLM client and the data are **injectable**, so unit tests run fully offline.

## How it works

1. **Load** (`recommend.load_catalog` + `recommend.load_profiles`): read the
   bundled `data/catalog.json` (~8 items) and `data/profiles.json` (~3 users).
2. **Rank** (`recommend.rank_items`): score each item by overlap with the
   profile's liked genres/tags, drop zero-overlap items, sort (score desc, id
   asc), take top-k.
3. **Explain** (`recommend.reason_for` → `llm.chat`): one `chat.completions` call
   per item for a grounded one-sentence reason. For qwen3 models we prepend
   `/no_think`.

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"09-recommendations"}`
- `POST /run` body `{"user_id": str, "k": int|null}` (default k=3) →
  `{"recommendations": [{"item_id": str, "title": str, "reason": str}]}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
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
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#            -d '{"user_id":"u1","k":3}'
```

## Feasibility note

Recommendations are a great raw-api fit: the ranking is deterministic arithmetic
that belongs in code, and the only language task — a short justification — is a
single constrained model call. No Anthropic model is needed; local Qwen writes
the reasons fine.
