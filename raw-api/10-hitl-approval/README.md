# UC10 — hitl-approval (raw-api)

A workflow that **pauses for human approval** before executing a high-risk
action. Built the **raw-api** way: the `openai` SDK pointed at the gateway, and
the pause/resume plumbing hand-written in `app/hitl.py` so you can see exactly
how the checkpoint is stored and reloaded.

## What it demonstrates

- **A human-in-the-loop checkpoint.** `POST /run` asks the LLM to draft a
  proposed action (a short refund-approval message), persists the paused run,
  and returns `status=awaiting_approval`. The workflow does not execute until a
  human calls `POST /resume`.
- **Manual state persistence.** Because raw-api has no native interrupt, we
  hand-build a thread-safe `CheckpointStore` (a dict keyed by `run_id`; a real
  system would use SQLite/Redis). `/resume` pops the run, continues, and the run
  can only terminate once.
- The LLM call is **injectable**, so unit tests run fully offline with a stub.

## How it works

1. **Draft + pause** (`hitl.start_run`): one `chat.completions` call drafts the
   action; we save a `PausedRun(run_id, request, proposed_action)` and return.
2. **Resume** (`hitl.resume_run`): look the run up by `run_id`, pop it (so it
   cannot resume twice), then — if approved — mark the action sent (executed);
   if not approved — return rejected (optionally echoing feedback).

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"10-hitl-approval"}`
- `POST /run` body `{"request": str}` →
  `{"run_id": str, "status": "awaiting_approval", "proposed_action": str}`
- `POST /resume` body `{"run_id": str, "approved": bool, "feedback": str|null}` →
  approved → `{"status":"executed","result": <action marked sent>}`;
  not approved → `{"status":"rejected","result": null}`;
  unknown `run_id` → **404**.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (must pass with no network):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
# then:
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"request":"Approve a $40 refund for a defective robot vacuum."}'
#   curl -X POST localhost:8000/resume -H 'content-type: application/json' \
#     -d '{"run_id":"<id from /run>","approved":true}'
```

## Why this is impractical in raw API

This use case is a poor fit for raw-api, and that is the point — compare it with
[`langgraph/10-hitl-approval`](../../langgraph/10-hitl-approval), the showcase.

A human-approval checkpoint requires **suspending a running workflow, durably
persisting its state, and resuming it later from exactly where it stopped**. The
raw-api version hand-builds every piece of that:

- **The checkpoint store** (`CheckpointStore`) — we manually persist the paused
  run (proposed action + original request) keyed by `run_id`, add locking for
  concurrency, and pop-on-terminal so a run can't resume twice. This is a
  hand-rolled checkpointer.
- **The resume entry point** — there is no "continue from where you paused";
  `/resume` is a separate request that has to look the state up and re-enter the
  logic itself. With a longer workflow you would also have to re-thread all the
  intermediate variables yourself.
- **No native pause signal** — the "pause" is just *return early and remember*.
  Any branching/looping around the checkpoint multiplies the bookkeeping.

LangGraph gives all of this for **free**: `interrupt()` suspends the graph at the
node, a checkpointer (`MemorySaver`) durably saves the entire graph state under
`thread_id == run_id`, and `Command(resume=...)` re-enters the graph at the exact
interrupt point with the human's decision. No store, no locks, no manual
re-threading. See the langgraph README for the side-by-side.
