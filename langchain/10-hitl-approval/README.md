# UC10 — hitl-approval (langchain)

A workflow that **pauses for human approval** before executing a high-risk
action, built with **langchain**: an LCEL draft chain plus a **manual
pause/resume workaround** around it.

## What it demonstrates

- **LangChain has no first-class interrupt.** An LCEL chain
  (`prompt | llm | StrOutputParser`) is built to run start-to-finish. To insert a
  human checkpoint you stop *outside* the chain and manage the pause yourself.
- **The workaround.** `POST /run` runs the draft chain to completion to get the
  proposed action, then explicitly stops: it stashes the run in a thread-safe
  `RunRegistry` keyed by `run_id` and returns `status=awaiting_approval`. The
  chain is not suspended — there is no graph to suspend — the app just holds the
  result and declines to execute. `POST /resume` looks the run up and runs the
  execute step, popping the run so it can only terminate once.
- The LLM is **injectable**, so unit tests run fully offline with a fake.

Contrast with [`langgraph/10-hitl-approval`](../../langgraph/10-hitl-approval),
where `interrupt()` + a checkpointer suspends and resumes the *actual workflow*
at the node. Here the "pause" is application-level bookkeeping the developer owns
— closer to the raw-api version than to the native langgraph one.

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"10-hitl-approval"}`
- `POST /run` body `{"request": str}` →
  `{"run_id": str, "status": "awaiting_approval", "proposed_action": str}`
- `POST /resume` body `{"run_id": str, "approved": bool, "feedback": str|null}` →
  approved → `{"status":"executed","result": <action marked sent>}`;
  not approved → `{"status":"rejected","result": null}`;
  unknown `run_id` → **404**.

## Streaming through the approval gate

`POST /run/stream` streams the draft and **ends at the gate** — its last frame is
`awaiting_approval`. `POST /resume/stream` opens a second stream once the human
decides. Holding one connection open across a human decision would make every
proxy timeout or restart a lost run; the checkpoint is the contract, not the
connection. `/resume/stream` emits no `token` frames on purpose: approving
executes the text the human already read.

Full contract and the four-way comparison of *what survives the gate*:
[`docs/streaming.md`](../../docs/streaming.md).

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen-local-instruct` | model alias (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (optional) |
| `LLM_MAX_TOKENS` | `256` | max generated tokens (optional) |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

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
