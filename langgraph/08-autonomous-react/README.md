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
  "stopped_reason": "final_answer"|"max_steps"}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `claude-haiku-4-5` | model alias (qwen3 → `/no_think` auto-applied) |
| `MAX_STEPS` | `6` | ReAct loop iteration cap |

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
```

## Model note (UC8-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen.
Reason: UC8 needs a reliable *multi-step* tool chain (search → 30 → calculator →
60). The free local model (`qwen-local-instruct`, qwen2.5-7B) was unreliable at
driving the two-tool text-ReAct cycle — it garbled the `Action`/`Action Input`
format, skipped the second tool, or answered "60" without actually using the
calculator. This is the documented "Haiku fallback" rule from the build spec
(mirrors UC3). The integration test is marked `anthropic`; unit tests remain
fully mocked/offline and require no key.
