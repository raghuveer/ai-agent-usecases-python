# UC10 — hitl-approval (langgraph) — showcase

A workflow that **pauses for human approval** before executing a high-risk
action. This is the showcase: LangGraph's `interrupt()` + a checkpointer is the
clean, native mechanism for exactly this, and the contrast with the hand-built
raw-api version is the point.

## The graph

```
draft ──► review (interrupt) ──► execute ──► END
```

- **draft** — the LLM drafts the proposed action (a short refund-approval
  message) from the request.
- **review** — calls `interrupt({...})`. This **suspends the graph**. A
  checkpointer (`MemorySaver`) durably saves the *entire* graph state keyed by
  the thread id. The proposed action is surfaced to the caller; nothing
  executes yet.
- **execute** — runs only after the human resumes. The decision arrives as the
  return value of `interrupt()`, delivered via `Command(resume=...)`: approved →
  the action is marked sent (executed); not approved → rejected.

## Why this is the native fit

Compare with [`raw-api/10-hitl-approval`](../../raw-api/10-hitl-approval), which
hand-builds a `CheckpointStore` (dict + locks + pop-on-terminal) and a separate
resume entry point that re-threads state by hand. LangGraph gives all of that for
free:

| Need | raw-api (hand-built) | langgraph (native) |
|---|---|---|
| Pause mid-workflow | return early + remember | `interrupt()` |
| Persist paused state | manual `CheckpointStore` (dict/SQLite + locks) | a checkpointer (`MemorySaver`) |
| Key the paused run | `run_id` dict key | `thread_id == run_id` |
| Resume from the pause | separate request re-enters the logic | `Command(resume=...)` re-enters at the interrupt |
| Run only once | pop-on-terminal | terminal state has no pending nodes |

`thread_id == run_id`: each `/run` picks a fresh `run_id` and invokes the graph
under that thread; `/resume` re-enters the **same** thread, so the checkpointer
restores the exact paused state. `MemorySaver` is fine for this example (in-proc,
ephemeral); swap in `SqliteSaver`/`PostgresSaver` for durability across restarts.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"10-hitl-approval"}`
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
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `256` | max generation tokens |

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
