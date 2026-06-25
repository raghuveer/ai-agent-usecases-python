# UC5 — support-triage (langgraph)

Triage is a router, which is the natural showcase for langgraph. The flow is a
compiled `StateGraph`:

```
            classify
               |
       (conditional edge on intent)
          /    |    \
    billing technical general     <- specialist responder nodes
          \    |    /
           finalize                <- sets escalate flag from confidence
               |
              END
```

- **classify** asks the LLM for `{intent, confidence}` JSON and writes both to
  graph state.
- a **conditional edge** routes to exactly one specialist node, each with its own
  system prompt, which generates the reply.
- **finalize** sets `escalate=true` when `confidence < ESCALATE_THRESHOLD` (0.5).
  (The node is named `finalize` because a langgraph node may not share a name
  with a state key, and `escalate` is a state field.)

A module-level dict gives per-session memory; prior turns feed the classifier and
responder. The LLM is injected so unit tests run offline.

## API

`GET /health` → `{"status":"ok","approach":"langgraph","usecase":"05-support-triage"}`

`POST /run`
```json
{"message": "I was double charged on my invoice", "session_id": "abc"}
```
→
```json
{"intent": "billing", "confidence": 0.92, "response": "...", "escalate": false}
```

## Setup & tests

```sh
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration
```

Copy `.env.example` to `.env` and set `LLM_GATEWAY_KEY`. Default model is the free
local alias `qwen-local-instruct`.
