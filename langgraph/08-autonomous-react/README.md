# UC8 — autonomous-react (langgraph) — showcase approach

A general **ReAct agent** that loops-until-done over a tiny tool registry,
choosing tools by what it observes. This is **LangGraph's native territory**: an
autonomous agent that cycles until it reaches a final answer. UC8 is the
showcase use case for langgraph, so the graph cycle is the centerpiece.

## The ReAct cycle (the whole point)

The agent is a `StateGraph` whose nodes form an explicit reason → act → observe
loop, joined by a **conditional edge** that either loops back or terminates:

```
            ┌───────────────────────────────────────────────┐
            │                                               │
            ▼                                               │
  ENTRY ─► reason ──route──tool──► act ──► observe ─────────┘
            │
            └────────route──(final_answer | max_steps)──► END
```

- **reason** — call the LLM with the running transcript; it emits either an
  `Action`/`Action Input` (text ReAct protocol) or a `Final Answer`.
- **route** (conditional edge `reason → {act, END}`) — if the model produced a
  Final Answer, or the step count has reached `max_steps`, go to `END`;
  otherwise go to `act`.
- **act** — dispatch the parsed tool from the `TOOLS` registry.
- **observe** — append `Observation: <result>` to the transcript, record the
  structured step, then the static back-edge `observe → reason` closes the loop.

The transcript and the structured `steps` live in **typed graph state**
(`ReactState`), and `messages` uses LangGraph's `add_messages` reducer so each
node appends to the running conversation.

## Text-based ReAct, not native function-calling

Provider-native `tools=` function-calling is inconsistent across this gateway,
so the agent drives ANY chat model with a strict **text** format:

```
Thought: <reasoning>
Action: <calculator | search>
Action Input: <single-line input>
```

…or, when finished, `Final Answer: <answer>`.

## Tools (deterministic, offline)

- **`calculator(expression)`** — safe arithmetic via `ast` (NOT `eval`); only
  numbers and `+ - * / // % **`. Hostile input like `__import__('os')` is
  rejected and surfaced to the model as an error observation.
- **`search(query)`** — keyword lookup over bundled `data/facts.md`.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"08-autonomous-react"}`
- `POST /run` body `{"task": str, "max_steps": int|null}` →
  `{"answer": str, "steps": [{thought, action, action_input, observation}],
  "stopped_reason": "final_answer"|"max_steps", "trace": object|null}`
- `POST /run?trace=1` — same, with `trace` populated. See below.

## See what it actually did (`?trace=1`)

The shared format ([`docs/trace-format.md`](../../docs/trace-format.md)) plus one
field only this approach can produce: **`graph_path`, the nodes actually
visited**. The other three have a call sequence; this one has a *route*.

A real run against `claude-haiku`:

```
spans      : llm chat · tool search · llm chat · tool calculator · llm chat
graph_path : reason → act → observe → reason → act → observe → reason
usage      : {input_tokens: 3469, output_tokens: 193}
```

The cycle is right there in the path — and an early exit shows up as its
absence: a task the model answers immediately traces as `graph_path: ["reason"]`
with zero tool calls. That is the structural claim of this approach made
checkable instead of asserted.

Getting it required one real detail: **callbacks must go in the graph's run
config, not on the model.** LangGraph reports the executing node via
`metadata["langgraph_node"]` on chain events, so a handler attached to the LLM
records every model call and none of the route. Hence `run_react(...,
callbacks=[...])`, which forwards them into `graph.invoke(config=...)`.

Token usage is identical to `raw-api/08` and `langchain/08` (3,469 / 193): the
graph changes the control flow, not the payload.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `384` | max generation tokens |
| `MAX_STEPS` | `6` | ReAct loop iteration cap |

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
```

## Model note (UC8-specific)

This use case defaults to **`claude-haiku`**, not the free local Qwen.
Reason: UC8 needs a reliable *multi-step* tool chain (search → 30 → calculator →
60). The free local model (`qwen-local-instruct`, qwen2.5-7B) was unreliable at
driving the two-tool text-ReAct cycle — it garbled the `Action`/`Action Input`
format, skipped the second tool, or answered "60" without actually using the
calculator. This is the documented "Haiku fallback" rule from the build spec
(mirrors UC3). The integration test is marked `anthropic`; unit tests remain
fully mocked/offline and require no key.

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
