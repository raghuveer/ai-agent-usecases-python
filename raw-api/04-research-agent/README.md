# UC4 — research-agent (raw-api)

A research agent that plans sub-queries, gathers evidence from a bundled local
corpus with a deterministic `search` tool, and synthesises a cited answer —
built the **raw-api** way. The `openai` SDK points at the gateway and the
**text-based ReAct loop is hand-written**, so you can see exactly what is sent,
how the model's `Action`/`Action Input` is parsed, and how observations are fed
back. Offline: NO web.

## What it demonstrates

- **Approach trade-off:** raw-api gives full, explicit control. There is no
  agent framework — `app/agent.py` builds the ReAct prompt, parses the last
  `Action` / `Action Input` with a regex, runs the tool, appends
  `Observation:`, and loops until `Final Answer:` or `max_steps`. The cost is
  that *you* write the loop, the parser, and the tool dispatch.
- **Text ReAct, not native function-calling.** The protocol is plain text so it
  works with any chat model and does not depend on the gateway's tool-call
  support.
- The LLM call is **injectable**, so unit tests run fully offline with scripted
  model outputs.

## How it works

1. **Corpus + tool** (`agent.Corpus`): `data/corpus/*.md` is split into
   paragraphs; `search(query)` scores them by keyword overlap and returns the
   top snippets with their `[source.md]` filenames. Deterministic, offline.
2. **ReAct prompt** (`agent.SYSTEM_PROMPT`): defines the one tool and the exact
   `Thought / Action / Action Input` → `Observation` → `Final Answer` format.
3. **Loop** (`agent.run_agent`): call LLM → `parse_step` (regex) → run `search`
   → append `Observation` → repeat. Collects the source filenames seen into
   `sources`. Stops on `Final Answer:` or `max_steps` (default 6). Parsing is
   defensive: missing fields tolerated; one nudge if neither Action nor Final
   Answer is present.

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"04-research-agent"}`
- `POST /run` body `{"question": str, "max_steps": int|null}` →
  `{"answer": str, "sources": [str], "steps": [{thought, action, action_input, observation}]}`

## Model choice — why this use case runs on Anthropic

Default per the build spec is the free **`qwen-local-instruct`**. We built on it
and ran the integration loop: **Qwen could not reliably drive the text ReAct
protocol.** It emitted prose paragraphs instead of `Action: search` /
`Action Input:`, pulled in outside knowledge, and never reached a `Final
Answer` — the loop hit `max_steps` every time. Per the spec's fallback rule this
use case therefore switches its default to **`claude-haiku`** (set in
`settings.py`, `.env`, and `.env.example`), and the integration test is marked
`@pytest.mark.anthropic` in addition to `@pytest.mark.integration`.

### Gateway PII-filter note

The gateway redacts the proper noun "Northwind" to a `<PERSON>` token on the way
in, which initially derailed the model (it refused, thinking the question
contained a placeholder). The loop now **desensitizes the brand name to a
neutral phrase in the text sent to the model** (the corpus, the user's question,
and the response keep the real name) and restores it in the final answer. With
that in place the agent reliably completes the loop and cites both
`returns.md` and `warranty.md`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `AGENT_MAX_STEPS` | `6` | ReAct step cap |
| `AGENT_TOP_K` | `3` | snippets per search |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live loop via the gateway (Anthropic; gated):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration
```

## Gateway note — `stop` sequences (fixed 2026-07-30)

A text ReAct loop must halt the model right after its `Action:` so it cannot
invent the `Observation:` — the loop supplies real tool output. The obvious way
is an OpenAI `stop` array, and that is what this project sent.

**The AI Utility Platform gateway returns `500 internal_error` for any request
carrying `stop` on a `claude-*` alias** — it does not translate `stop` into
Anthropic's `stop_sequences`. The same request succeeds without `stop`, and the
Ollama-backed aliases honour `stop` normally. Since this use case defaults to
`claude-haiku`, every live run failed on its first call.

The fix treats the cut as *our* invariant rather than the server's favour:

- `model_profile()` carries a `supports_stop` capability (False for `claude-*`),
  so `stop` is sent only to endpoints that accept it.
- `truncate_at_stop()` cuts the reply at the first `Observation:` either way.

This is the more portable arrangement regardless of the gateway bug: `stop` is
advisory, several providers ignore it, and a model that writes its own
`Observation:` would otherwise be feeding itself fabricated tool output.
