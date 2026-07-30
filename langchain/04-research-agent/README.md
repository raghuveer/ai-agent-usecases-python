# UC4 — research-agent (langchain)

A research agent that plans sub-queries, gathers evidence from a bundled local
corpus with a deterministic `search` tool, and synthesises a cited answer —
built with **LangChain primitives**. The `search` function is wrapped as a
LangChain `Tool`, and each ReAct turn is produced by an LCEL chain
(`ChatPromptTemplate | llm | StrOutputParser`). The protocol is plain **text
ReAct**, not provider-native function-calling. Offline: NO web.

## What it demonstrates

- **Approach trade-off:** LangChain gives you reusable building blocks — a real
  `Tool` object (`agent.make_search_tool`) and an LCEL chain for the per-step
  model call (`agent.build_turn_chain`). The ReAct control loop itself is still
  explicit so the text protocol is robust on this gateway; the langchain value
  is the composable `Tool` + chain, and the injectable `BaseChatModel`.
- **Text ReAct, not native function-calling** — works with any chat model.
- The model is injected, so unit tests use `FakeListChatModel` and run offline.

## How it works

1. **Corpus + tool** (`agent.Corpus`, `agent.make_search_tool`): `data/corpus/*.md`
   is split into paragraphs; `search(query)` scores by keyword overlap and is
   exposed as a LangChain `Tool` returning snippets with `[source.md]` tags.
2. **Per-turn chain** (`agent.build_turn_chain`): `ChatPromptTemplate | llm |
   StrOutputParser` renders the system + transcript and returns one model turn.
3. **Loop** (`agent.run_agent`): invoke the chain → `parse_step` (regex) → run
   the `Tool` → append `Observation` → repeat until `Final Answer:` or
   `max_steps` (default 6). Sources seen are collected from the tool into
   `sources`. Parsing is defensive (tolerates missing fields / a redacted
   `Action Input` label; nudges once if neither Action nor Final Answer).

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"04-research-agent"}`
- `POST /run` body `{"question": str, "max_steps": int|null}` →
  `{"answer": str, "sources": [str], "steps": [{thought, action, action_input, observation}]}`

## Model choice — why this use case runs on Anthropic

Default per the build spec is the free **`qwen-local-instruct`**. On the live
loop **Qwen could not reliably drive the text ReAct protocol** — it emitted
prose instead of `Action: search` / `Action Input:`, used outside knowledge, and
never reached a `Final Answer` (the loop hit `max_steps`). Per the spec's
fallback rule this use case switches its default to **`claude-haiku`** (set
in `settings.py`, `.env`, `.env.example`), and the integration test is marked
`@pytest.mark.anthropic` as well as `@pytest.mark.integration`.

### Gateway PII-filter note

The gateway redacts the proper noun "Northwind" to a `<PERSON>` token on input,
which initially made the model refuse. The loop **desensitizes the brand name to
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
| `AGENT_MAX_STEPS` | `6` | ReAct step cap |
| `AGENT_TOP_K` | `3` | snippets per search |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (optional) |
| `LLM_MAX_TOKENS` | `512` | max generated tokens (optional) |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

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
