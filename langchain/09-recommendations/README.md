# UC9 — recommendations (langchain)

Content recommendations over a tiny bundled catalog, built with **langchain**:
the deterministic ranking is plain Python, and a small **LCEL chain**
(`prompt | llm | StrOutputParser`) writes a one-sentence reason per recommended
item.

## What it demonstrates

- **Determinism where it matters:** scoring/ranking is reproducible plain Python
  (`recommend.rank_items`) — overlap between the catalog item's genres/tags and
  the user's liked genres/tags. The model never picks the items.
- **LLM only for the "why":** `recommend.build_reason_chain` is a tiny LCEL chain
  invoked once per top-k item, constrained to the supplied profile/item data.
- The LLM and the data are **injectable**, so unit tests run fully offline with
  `FakeListChatModel`.

## How it works

1. **Load** (`load_catalog` + `load_profiles`): read `data/catalog.json` (~8
   items) and `data/profiles.json` (~3 users).
2. **Rank** (`rank_items`): score each item by overlap with the profile's liked
   genres/tags, drop zero-overlap items, sort (score desc, id asc), take top-k.
3. **Explain** (`build_reason_chain`): `prompt | llm | StrOutputParser` per item
   for a grounded one-sentence reason. For qwen3 models we prepend `/no_think`.

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"09-recommendations"}`
- `POST /run` body `{"user_id": str, "k": int|null}` (default k=3) →
  `{"recommendations": [{"item_id": str, "title": str, "reason": str}]}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `REC_TOP_K` | `3` | default number of recommendations |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (optional) |
| `LLM_MAX_TOKENS` | `128` | max generated tokens (optional) |

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

Recommendations fit langchain cleanly: ranking is deterministic arithmetic in
plain Python, and the only language task — a short justification — is a one-line
LCEL chain. No Anthropic model is needed; local Qwen writes the reasons fine.
