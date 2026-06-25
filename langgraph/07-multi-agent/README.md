# UC7 — multi-agent (langgraph) — the showcase

An **orchestrator** routes work between three specialised sub-agents over one
**shared, typed state**. This is the use case LangGraph is built for, so it is
the showcase approach.

```
researcher ──► writer ──► reviewer ──► (route) ──approved──► END
                 ▲                        │
                 └──────reject (revise)───┘   (capped at MAX_REVISIONS)
```

- **researcher** node — deterministic, offline `research()` over
  `data/corpus/*.md`. Writes `research` into the shared state.
- **writer** node — role-prompted LLM. Reads `research` from state and drafts a
  short summary (and, on a revise pass, addresses the reviewer's critique).
  Writes `draft`.
- **reviewer** node — role-prompted LLM. Reads `draft`, writes `review` and the
  `approved` flag.
- **route** — a **conditional edge**: approved (or revision cap hit) → END,
  otherwise loop back through `revise` → `writer`.

## Why this is the native fit

Compare the `raw-api/07-multi-agent` sibling, where the orchestration is a
hand-written `while` loop threading local variables between functions. Here:

- **Each sub-agent is a graph node** — a first-class unit, not just a named
  function. The orchestrator is the compiled graph.
- **One shared typed state** (`MultiAgentState`) carries the research notes,
  draft, critique, approval flag, and revision count. No node hands data to the
  next by hand: each reads the slice it needs and writes its own back.
- **Routing is one declarative conditional edge.** "reviewer → (revise | END)"
  and the revision cap are expressed as graph structure, not imperative control
  flow. The revise loop is a real cycle (`revise → writer`).
- **Adding a sub-agent is adding a node + an edge**, not re-threading variables.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"07-multi-agent"}`
- `POST /run` body `{"topic": str}` →
  `{"draft": str, "review": str, "approved": bool, "contributions": {"research", "writer", "reviewer"}}`

## The corpus (offline researcher)

`data/corpus/*.md` holds neutral topic notes (tidal energy, vertical farming,
mesh networking). Neutral topic names are deliberate: the gateway's PII filter
masks proper nouns/brands on the wire, so a branded corpus risks key terms being
redacted. The researcher scores each bullet line by word overlap with the topic
and returns the best `RESEARCH_TOP_K`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku-4-5` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `400` | max generation tokens |
| `RESEARCH_TOP_K` | `4` | corpus facts the researcher gathers |
| `MAX_REVISIONS` | `1` | reviewer reject → writer revise cap |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the gateway running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#   -d "{\"topic\":\"tidal energy\"}"
```

## Model note (UC7-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen.
Reason: UC7 needs reliable *role-following* — the writer must draft only from the
notes, and the reviewer must emit a parseable `APPROVED:` verdict — which the
free local model could not do reliably. The integration test therefore spends a
small, capped amount of Anthropic budget and is marked `anthropic`; unit tests
remain fully mocked/offline and require no key.
