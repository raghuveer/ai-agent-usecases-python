# UC5 — support-triage (raw-api)

Classify a support message into an intent, route it to a per-intent specialist,
generate a reply, and escalate low-confidence cases — built with the plain
`openai` SDK against the gateway. Simple in-memory session memory keys the
running conversation by `session_id`.

## Flow

1. **Classify** — the LLM returns `{"intent", "confidence"}` JSON.
2. **Route** — pick the specialist system prompt for `billing` / `technical` /
   `general`.
3. **Respond** — the LLM writes the reply with that specialist prompt.
4. **Escalate** — `escalate=true` when `confidence < ESCALATE_THRESHOLD` (0.5).

Prior turns of a session are fed back into both the classifier and responder.

## API

`GET /health` → `{"status":"ok","approach":"raw-api","usecase":"05-support-triage"}`

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

# Unit tests (offline, fully mocked):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Integration (free local Qwen via the gateway):
RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration
```

Copy `.env.example` to `.env` and set `LLM_GATEWAY_KEY`. The default model is the
free local alias `qwen-local-instruct`.

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.
