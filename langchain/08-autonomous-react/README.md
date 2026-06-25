# UC8 — autonomous-react (langchain)

A general **ReAct agent** that loops-until-done over a tiny tool registry,
choosing tools by what it observes. Built the **langchain** way: tools are
LangChain `Tool` objects and the agent is assembled from LangChain primitives,
but it is driven by a strict **text** ReAct protocol rather than provider-native
function-calling.

## What it demonstrates

- **Text-based ReAct, not native function-calling.** Provider-native tool-calling
  (`create_react_agent` + `AgentExecutor`) is inconsistent across this gateway,
  so we keep the tools as LangChain `Tool` objects but drive them with the same
  portable text format every model can emit.
- **Approach trade-off:** langchain gives you reusable `Tool` abstractions and a
  `BaseChatModel` interface (so `FakeListChatModel` makes offline testing
  trivial), at the cost of a bit of framework indirection around the loop in
  `app/react.py`.

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

`run_react` invokes the chat model with a LangChain message list, parses the
last `Action`/`Action Input`, invokes the matching `Tool`, threads back
`Observation: <result>`, and loops. Stops on `Final Answer:` or `max_steps`
(default 6).

## Tools (deterministic, offline)

- **`calculator(expression)`** — safe arithmetic via `ast` (NOT `eval`); only
  numbers and `+ - * / // % **`. Hostile input like `__import__('os')` is
  rejected and surfaced to the model as an error observation.
- **`search(query)`** — keyword lookup over bundled `data/facts.md`.

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"08-autonomous-react"}`
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
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (optional) |
| `LLM_MAX_TOKENS` | `384` | max generated tokens (optional) |

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
```

## Model note (UC8-specific)

This use case defaults to **`claude-haiku-4-5`**, not the free local Qwen.
Reason: UC8 needs a reliable *multi-step* tool chain (search → 30 → calculator →
60). The free local model (`qwen-local-instruct`, qwen2.5-7B) was unreliable at
driving the two-tool text-ReAct loop — it garbled the `Action`/`Action Input`
format, skipped the second tool, or answered "60" without actually using the
calculator. This is the documented "Haiku fallback" rule from the build spec
(mirrors UC3). The integration test is marked `anthropic`; unit tests remain
fully mocked/offline and require no key.
