# UC4 — research-agent (langgraph)

A research agent that plans sub-queries, gathers evidence from a bundled local
corpus with a deterministic `search` tool, and synthesises a cited answer —
modelled as a **LangGraph `StateGraph` cycle**. The ReAct loop is the graph: the
whole transcript and collected sources live in typed state. The protocol is
plain **text ReAct**, not provider-native function-calling. Offline: NO web.

## The graph (this is the point of the langgraph approach)

```
        ┌──────────────────────────────┐
        ▼                              │
   reason ──route──► act (search) ─────┘
     │       │
     │       ├──► cap ──► END     (step_count >= max_steps)
     │       └──► END            (Final Answer parsed, or stuck)
```

- **`reason`** calls the model for one turn and parses it into scratch state
  (`pending_action` / `pending_input`, or a `final_answer`).
- **`route`** is the conditional edge: `END` on a final answer, `cap` when the
  step budget is exhausted, otherwise `act` (and it loops back to `reason` after
  a one-time nudge if the model emitted neither an Action nor a Final Answer).
- **`act`** runs `search`, appends the `Observation` to the transcript, records
  the step, dedupes sources, and increments `step_count`. The edge `act →
  reason` closes the cycle.

State (`AgentState`, a `TypedDict`) threads `question`, `transcript`, `steps`,
`sources`, `step_count`/`max_steps`, and the per-cycle scratch fields.

## How it works

1. **Corpus + tool** (`agent.Corpus`): `data/corpus/*.md` split into paragraphs;
   `search(query)` scores by keyword overlap, returns snippets with `[source.md]`
   tags. Deterministic, offline.
2. **Graph** (`agent.build_agent_graph`): the cycle above, with the chat model
   injected (unit tests pass `FakeListChatModel`).
3. **Parsing** (`agent.parse_step`): regex over `Thought / Action / Action Input`
   and `Final Answer`; defensive (tolerates missing fields and a redacted
   `Action Input` label).

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"04-research-agent"}`
- `POST /run` body `{"question": str, "max_steps": int|null}` →
  `{"answer": str, "sources": [str], "steps": [{thought, action, action_input, observation}]}`

## Model choice — why this use case runs on Anthropic

Default per the build spec is the free **`qwen-local-instruct`**. On the live
loop **Qwen could not reliably drive the text ReAct protocol** — it emitted prose
instead of `Action: search` / `Action Input:`, used outside knowledge, and never
reached a `Final Answer` (the graph hit its `cap` node). Per the spec's fallback
rule this use case switches its default to **`claude-haiku`** (set in
`settings.py`, `.env`, `.env.example`), and the integration test is marked
`@pytest.mark.anthropic` as well as `@pytest.mark.integration`.

### Gateway PII-filter note

The gateway redacts the proper noun "Northwind" to a `<PERSON>` token on input,
which initially made the model refuse. The graph **desensitizes the brand name to
a neutral phrase in the text sent to the model** (the corpus, the question, and
the response keep the real name) and restores it in the final answer. With that
in place the agent reliably completes and cites both `returns.md` and
`warranty.md`.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `512` | max generation tokens |
| `AGENT_MAX_STEPS` | `6` | ReAct step cap |
| `AGENT_TOP_K` | `3` | snippets per search |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live graph via the gateway (Anthropic; gated):
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
