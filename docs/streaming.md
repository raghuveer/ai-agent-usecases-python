<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Raghuveer Dendukuri -->

# Streaming (server-sent events)

`POST /run/stream` returns `text/event-stream` and reports the agent's work as it
happens, rather than after it finishes. A ReAct loop is the case where this
matters most: you watch it reason, call a tool, read the result, and go again.

Implemented on the **UC07** and **UC08** projects. Same event contract in all of
them, so the four approaches can be compared frame for frame — the companion to
[`trace-format.md`](trace-format.md), which records the same run after the fact.

## Events

```
event: token
data: {"text": "Thought"}

event: thought
data: {"text": "I need to look up the return window"}

event: step
data: {"step": {"thought": "…", "action": "search",
                "action_input": "return window", "observation": "[returns.md] 30 days"}}

event: final
data: {"answer": "60 days", "stopped_reason": "final_answer", "steps": [ … ]}

event: error
data: {"message": "…"}
```

| Event | When |
|---|---|
| `token` | incremental model text, as it arrives |
| `thought` | the turn's parsed reasoning line |
| `step` | a tool ran; carries the observation |
| `final` | always last on success — the same payload `POST /run` returns |
| `error` | the run failed; **always framed, never silent** |

A stream that dies without a frame is indistinguishable from one that finished,
so failures are events too. Frames end with a blank line — omit it and the client
waits forever for a frame that already arrived.

## All four, measured

Same task, same machine, real `uvicorn` processes; the three OpenAI-surface
approaches against local `qwen3:1.7b`, the Agent SDK against `claude-haiku`
(the gateway cannot stream — see below).

| | raw-api | langchain | langgraph | claude-agent-sdk |
|---|---|---|---|---|
| `token` frames | 53 | 52 | 55 | **0 — cannot** |
| `node` frames | — | — | **4** | — |
| Other frames | thought, step, final | thought, step, final | step, final | turn ×3, step ×2, final |
| **Time to first frame** | 0.54s | 0.75s | **0.36s** | **5.49s** |
| Total spread | 4.13s | 3.60s | 3.75s | 3.47s |
| How you stream | `stream=True`, parse deltas yourself | `llm.stream()` → `AIMessageChunk` | `stream_mode=["messages","updates"]` | `async for` over SDK messages |

Two results worth sitting with:

**Only langgraph can stream a route.** `event: node` frames arrive live —
`reason → act → observe → reason` — because the framework knows what a node is.
That is the running counterpart to `graph_path` in the trace, and the other three
have nothing to report it with.

**The Agent SDK cannot stream tokens at all, and you can feel it.** It yields
whole `AssistantMessage`s, so the first frame lands at **5.49s** against
**0.36s** for langgraph — roughly fifteen times longer before a user sees
anything. The harness owns the model call and exposes no deltas. This is the same
asymmetry the trace found (no per-call messages, no token counts), showing up
here as latency instead of missing fields: **writing no loop costs you the view
inside it.**

## Three things streaming forces you to confront

**1. Thinking tags arrive split across chunks.** qwen3 emits `<think>…</think>`
even under `/no_think` (see `TRACKING.md` finding 13). In a complete reply you can
regex it out. In a stream the tag comes as `"<th"` + `"ink>"`, and once you have
forwarded reasoning to the client you cannot take it back. Every approach here
runs a small incremental filter that holds back anything after a `<` until it
either completes a tag or proves not to be one — a few characters of latency in
exchange for a guarantee. A real run bears this out: the **first delta Ollama
sends is literally `<think>`**.

**1b. A plain function in an LCEL chain silently disables streaming.** LangChain
wraps a bare callable in a `RunnableLambda`, which must materialise its whole
input before it runs. So this:

```python
prompt | llm | StrOutputParser() | strip_thinking     # looks harmless
```

turns `chain.stream()` into a single chunk per chain — **measured: 2 token frames
instead of 149**, with no error and no warning. The endpoint still "streams"; it
just has nothing to stream.

A *generator* function is treated as a transform instead — it receives the
upstream iterator and yields as it goes, so streaming survives:

```python
prompt | llm | StrOutputParser() | strip_thinking_stream   # generator
```

`invoke()` still works either way; LangChain drains the generator. This is worth
knowing before you add any post-processing step to a chain you intend to stream.

**2. Your test client probably buffers.** FastAPI's `TestClient` collected all 56
frames and delivered them at once — first frame and last frame at the same
timestamp. The endpoint was fine; the ASGI test transport was not. Verified
against a real `uvicorn` process instead:

```
frames      : 56
first frame : 0.54s
last frame  : 4.68s
spread      : 4.13s   <- genuinely incremental
```

If you are testing SSE and every frame lands together, suspect the client before
the server.

**3. Not every endpoint supports it.** Measured 2026-07-31 against the AI Utility
Platform gateway:

| Path | `stream: true` |
|---|---|
| Ollama direct (`:11434/v1`) | ✅ works |
| Gateway → Anthropic (`claude-*`) | ❌ `500 internal_error` |
| Gateway → Ollama (`qwen-local-*`) | ❌ empty response body |

**The gateway does not pass streaming through at all.** That is a platform
limitation, not a bug in these examples — the same request streams correctly when
pointed straight at Ollama. It is the second such finding after the `stop` array
(`TRACKING.md` finding 11), and it has a practical consequence: to see streaming
work, point `LLM_BASE_URL` at Ollama or a provider directly.

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_GATEWAY_KEY=x LLM_MODEL=qwen3:1.7b \
  .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

curl -N -X POST localhost:8000/run/stream \
     -H 'content-type: application/json' -d '{"task":"return window doubled?"}'
```

`curl -N` disables buffering. Without it you get the same "everything at once"
illusion as the test client.
