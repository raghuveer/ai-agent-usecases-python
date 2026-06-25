# UC8 — autonomous-react (raw-api)

A general **ReAct agent** that loops-until-done over a tiny tool registry,
choosing tools by what it observes. Built the **raw-api** way: the `openai` SDK
pointed at the gateway, and the entire agentic loop hand-written in
`app/react.py` so you can see exactly what is sent and how tool calls are
parsed and dispatched.

## What it demonstrates

- **Text-based ReAct, not native function-calling.** Provider-native `tools=`
  function-calling is inconsistent across this gateway, so we drive ANY chat
  model with a strict TEXT protocol the model emits as plain text. The loop
  parses it with regex.
- **Approach trade-off:** raw-api gives full, explicit control. There is no
  hidden agent executor — `react.run_react` calls the LLM, parses the last
  `Action`/`Action Input`, runs the tool, appends `Observation:`, and loops. The
  cost is that *you* write the parser, the dispatch, and the stop conditions.
- The LLM call is **injectable**, so unit tests script replies and run fully
  offline.

## The ReAct protocol

The system prompt instructs the model to emit, each turn:

```
Thought: <reasoning>
Action: <calculator | search>
Action Input: <single-line input>
```

…or, when finished:

```
Thought: <reasoning>
Final Answer: <answer>
```

The loop appends `Observation: <tool result>` after each tool call and stops on
`Final Answer:` or when `max_steps` (default 6) is reached.

## Tools (deterministic, offline)

- **`calculator(expression)`** — safe arithmetic via `ast` (NOT `eval`). The
  expression is parsed and a whitelisted AST walker evaluates only numbers and
  `+ - * / // % **`. Hostile input like `__import__('os')` raises
  `UnsafeExpression` and is surfaced to the model as an error observation.
- **`search(query)`** — keyword lookup over the bundled `data/facts.md` (a few
  `key: value` Northwind facts). Returns the best-matching line(s).

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"08-autonomous-react"}`
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
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#   -d "{\"task\":\"Find Northwind's return window in days, then double it.\"}"
```

## Model note (UC8-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen.
Reason: UC8 needs a reliable *multi-step* tool chain (search → 30 → calculator →
60). The free local model (`qwen-local-instruct`, qwen2.5-7B) was unreliable at
driving the two-tool text-ReAct loop — it garbled the `Action`/`Action Input`
format, skipped the second tool, or answered "60" without actually using the
calculator. This is the documented "Haiku fallback" rule from the build spec
(mirrors UC3). The integration test therefore spends a small, capped amount of
Anthropic budget and is marked `anthropic`; unit tests remain fully
mocked/offline and require no key.
