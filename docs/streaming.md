<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Raghuveer Dendukuri -->

# Streaming (server-sent events)

`POST /run/stream` returns `text/event-stream` and reports the agent's work as it
happens, rather than after it finishes. A ReAct loop is the case where this
matters most: you watch it reason, call a tool, read the result, and go again.

Implemented on the **UC08** projects. Same event contract in all of them, so the
four approaches can be compared frame for frame — the companion to
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

## Three things streaming forces you to confront

**1. Thinking tags arrive split across chunks.** qwen3 emits `<think>…</think>`
even under `/no_think` (see `TRACKING.md` finding 13). In a complete reply you can
regex it out. In a stream the tag comes as `"<th"` + `"ink>"`, and once you have
forwarded reasoning to the client you cannot take it back. Every approach here
runs a small incremental filter that holds back anything after a `<` until it
either completes a tag or proves not to be one — a few characters of latency in
exchange for a guarantee. A real run bears this out: the **first delta Ollama
sends is literally `<think>`**.

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
