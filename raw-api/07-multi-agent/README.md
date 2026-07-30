# UC7 — multi-agent (raw-api)

An **orchestrator** delegates to three specialised sub-agents and aggregates
their work into one response:

- **researcher** — gathers bullet facts about the topic by a *deterministic,
  offline* keyword search over `data/corpus/*.md` (no network, no LLM, no
  embeddings).
- **writer** — a role-prompted LLM call that drafts a short summary from the
  research notes.
- **reviewer** — a role-prompted LLM call that returns a critique plus an
  `APPROVED: yes/no` verdict.
- **revise loop** — if the reviewer rejects, the writer redrafts once (capped at
  `MAX_REVISIONS`) using the critique, then the reviewer judges again.

Built the **raw-api** way: the `openai` SDK pointed at the gateway, and the
entire orchestration hand-written in `app/agents.py` — you can see exactly which
sub-agent runs when, what each is sent, and how their outputs are threaded
together.

## ⚠️ Why this is impractical in raw API

This is the use case where raw API stops being a clean choice. Coordinating
sub-agents is exactly what a graph framework gives you for free, and here we
hand-roll all of it:

- **No sub-agent primitive.** Each "agent" is just a function plus a role
  prompt. There is no node abstraction, no registry, no isolation — only naming
  discipline keeps the researcher/writer/reviewer separate.
- **No shared state.** The research notes, the draft, the critique, the
  approval flag, and the revision count are plain local variables that *we*
  thread from one call to the next by hand. Add a fourth agent and you re-wire
  the threading manually.
- **Routing is a hand-written `while` loop.** The reject → revise decision, the
  revision cap, and "which agent runs next" are all imperative Python. There is
  no declarative edge that says "reviewer → writer on reject, else END"; we
  encode it and must keep the bound correct ourselves.
- **Aggregation is manual.** We assemble the `contributions` dict by hand at the
  end; nothing collects per-agent output for us.

The same workflow in **`langgraph/07-multi-agent`** is the native fit: each
sub-agent is a graph **node**, the transcript lives in one **shared typed
state**, and a **conditional edge** expresses reviewer → (writer | END) with the
revise loop as a real cycle. See that folder for the showcase. This raw-api
version still **works** — it is here to make the contrast concrete.

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"07-multi-agent"}`
- `POST /run` body `{"topic": str}` →
  ```json
  {
    "draft": "…",
    "review": "Critique: …\nAPPROVED: yes",
    "approved": true,
    "contributions": { "research": "- …", "writer": "…", "reviewer": "…" }
  }
  ```

## The corpus (offline researcher)

`data/corpus/*.md` holds a few neutral topic notes (tidal energy, vertical
farming, mesh networking). Neutral topic names are deliberate: the gateway's PII
filter masks proper nouns/brands on the wire, so a Northwind-style branded
corpus risks getting key terms redacted to `<PERSON>`/`<LOCATION>`. The
researcher scores each bullet line by word overlap with the topic and returns
the best `RESEARCH_TOP_K`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `RESEARCH_TOP_K` | `4` | corpus facts the researcher gathers |
| `MAX_REVISIONS` | `1` | reviewer reject → writer revise cap |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

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

This use case defaults to **`claude-haiku`**, not the free local Qwen.
Reason: UC7 needs reliable *role-following* — the writer must draft only from the
notes, and the reviewer must emit a parseable `APPROVED:` verdict. The free
local model (`qwen-local-instruct`, qwen2.5-7B) was unreliable at staying in
role and at the verdict format. The integration test therefore spends a small,
capped amount of Anthropic budget and is marked `anthropic`; unit tests remain
fully mocked/offline and require no key.
