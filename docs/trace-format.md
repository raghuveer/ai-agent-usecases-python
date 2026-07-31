<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Raghuveer Dendukuri -->

# Trace format v1

A uniform record of *what an agent actually did* — the exact messages sent to the
model, every tool call and its result, tokens, timings, and how the run ended.

Two reasons it exists:

1. **Learning.** The examples otherwise answer questions from a black box. A trace
   makes the same question, run four ways, directly comparable — you can see that
   the raw-api version sends one flat message list, the LangGraph version walks a
   graph, and the Agent SDK version never shows you its loop at all.
2. **Comparison.** Later, runs from these examples should be comparable against
   agents built with *other* frameworks. That requires a stable, boring format.

## The design rule: own the format, borrow the names

The repo owns the schema. No project depends on a tracing vendor, a server, or a
key — a trace is built with the standard library and returned as JSON. That keeps
every example clonable and air-gap-runnable, which is the whole point of the repo.

But the **field names follow the [OpenTelemetry GenAI semantic
conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)** — `gen_ai.system`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, and so on.

That is deliberate, and it is the answer to "should we just use Langfuse?".

- OTel GenAI is what other frameworks already emit. LangChain, LlamaIndex,
  Pydantic AI, and the OpenAI/Anthropic SDKs all have OTel instrumentation, so a
  future cross-framework comparison has a common vocabulary.
- **Langfuse, Arize Phoenix, Jaeger, and Grafana all ingest OTel.** Naming the
  fields this way makes any of them a ~30-line exporter rather than a re-design.
- Adopting a vendor's schema instead would mean re-shaping the data the first time
  we wanted a different one.

So: vendor-neutral today, one small adapter away from any viewer tomorrow.

## Shape

```jsonc
{
  "schema_version": 1,
  "run_id": "4f1c…",                  // uuid4
  "started_at": "2026-07-30T14:02:11Z",
  "duration_ms": 4120,
  "approach": "raw-api",              // raw-api | langchain | langgraph | claude-agent-sdk
  "usecase": "08-autonomous-react",
  "gen_ai": {
    "system": "openai",               // the wire protocol, per OTel: openai | anthropic
    "request": { "model": "claude-haiku", "temperature": 0.0, "max_tokens": 384 },
    "usage":   { "input_tokens": 1204, "output_tokens": 233 }   // run totals
  },
  "outcome": {
    "status": "ok",                   // ok | capped | error
    "stop_reason": "final_answer",    // final_answer | max_steps | max_turns | max_budget | max_tokens | error
    "steps": 3,
    "tool_calls": 2,
    "cost_usd": null                  // null when the endpoint does not report cost
  },
  "spans": [
    {
      "seq": 1,
      "type": "llm",                  // llm | tool
      "name": "chat",
      "duration_ms": 980,
      "gen_ai": { "usage": { "input_tokens": 402, "output_tokens": 61 } },
      "request": {
        "messages": [ { "role": "system", "content": "…" } ],  // EXACTLY what was sent
        "stop": ["Observation:"]
      },
      "response": { "content": "Thought: …\nAction: search" }
    },
    {
      "seq": 2,
      "type": "tool",
      "name": "search",
      "duration_ms": 3,
      "request":  { "input": "return window" },
      "response": { "content": "[returns.md] 30-day return window" }
    }
  ]
}
```

`cost_usd` is `null`, not `0.0`, when unknown. An OpenAI-compatible endpoint
reports token usage but not price; only the Agent SDK reports real cost. Reporting
a confident `0.0` for an unpriced run would be a lie, and the runs where it matters
most are the expensive ones.

`stop_reason` is a **shared** vocabulary, not a passthrough. Three approaches own
their loop and naturally speak it — `final_answer`, `max_steps`. The Agent SDK
does not: it reports Anthropic's own reasons, `end_turn` and `max_tokens`. Those
are translated on the way into the trace (`trace.to_trace_reason`), because a
comparison table whose rows say `final_answer` in three approaches and `end_turn`
in the fourth is comparing spelling, not behaviour. The untranslated value is
still available — each Agent SDK project returns it as `stop_reason` in its own
HTTP response, where being faithful to the SDK is the more useful choice.

## Where it is implemented

`?trace=1` is available on the **UC07** and **UC08** projects, all four approaches.
The rest still return only their answer.

## Sinks

Set by env, defaulting to the least surprising thing:

| `TRACE_SINK` | Behaviour |
|---|---|
| `none` *(default)* | Nothing is written to disk. `?trace=1` still returns the trace inline. |
| `file` | Each run writes `traces/<run_id>.json`, and appends one summary line to `traces/runs.jsonl`. |

`?trace=1` on `POST /run` returns the trace in the response regardless of sink.
That is the "see what happened" path, and it costs nothing to leave available.

### Why also a `runs.jsonl`

Per-run JSON is for reading one run. Comparing runs wants one row each:

```jsonc
{"run_id":"4f1c…","ts":"…","approach":"raw-api","usecase":"08-autonomous-react",
 "model":"claude-haiku","status":"ok","stop_reason":"final_answer","steps":3,
 "tool_calls":2,"input_tokens":1204,"output_tokens":233,"cost_usd":null,"duration_ms":4120}
```

Concatenate those across approaches — or later across repos and frameworks — and
you have something you can actually aggregate, with no database involved.

## What it shows, with all four implemented (UC08, measured)

Same use case, `claude-haiku`, traced through each approach:

| | raw-api | langchain | langgraph | claude-agent-sdk |
|---|---|---|---|---|
| Model calls visible | 3 | 3 | 3 | ✗ (4 *turns*) |
| Exact messages sent | ✅ | ✅ | ✅ | ✗ harness-built |
| Tool results | ✅ | ✅ | ✅ | ✗ harness-internal |
| Per-call latency | ✅ | ✅ | ✅ | ✗ |
| Input / output tokens | 3469 / 193 | 3469 / 193 | 3469 / 193 | ✗ not reported |
| Real cost | ✗ unpriced | ✗ unpriced | ✗ unpriced | ✅ $0.42278 |
| Graph route | — | — | ✅ `reason→act→observe→…` | — |
| How it attaches | hand-instrumented | callback handler | callbacks in graph config | read off the result |

Three findings fall straight out of that table:

1. **The three framework-free-ish approaches send byte-identical payloads.**
   3,469 / 193 across raw-api, langchain, and langgraph — LangChain and LangGraph
   add no prompt overhead for this use case. That is usually assumed, rarely
   measured.
2. **Visibility is inversely proportional to how much loop the framework wrote
   for you.** The Agent SDK needs the least code and can tell you the least.
   Nulls, not zeros, are how the format says so.
3. **Only the SDK can report real cost** — and that run cost **$0.42** against a
   few tenths of a cent for the others on the same model, because the harness
   prompt is re-paid every turn with no caching through the gateway. (Different
   task phrasing, so not a controlled benchmark — but the order of magnitude is
   consistent with the code-gen measurements in the root README.)

## Privacy

**A trace contains the full prompt, which contains the user's input.** Treat
`traces/` as sensitive:

- `TRACE_SINK` defaults to `none`, so nothing is persisted unless asked for.
- `TRACE_INCLUDE_PROMPTS=0` records span metadata, timings, and token counts while
  omitting message content and tool output.
- `traces/` is git-ignored.

The `?trace=1` response path is the safer default for learning: the caller sees
their own prompt echoed back, and nothing is stored.
