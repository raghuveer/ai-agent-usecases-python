# UC5 — support-triage (langchain)

Classify a support message into an intent, route it to a per-intent specialist,
generate a reply, and escalate low-confidence cases — built with langchain LCEL
chains against the gateway. Simple in-memory session memory keys the running
conversation by `session_id`.

## Flow

1. **Classify** — `classifier_prompt | llm | StrOutputParser`; parse the JSON
   into `{intent, confidence}`.
2. **Route** — dict lookup picks the specialist prompt for `billing` /
   `technical` / `general`.
3. **Respond** — `specialist_prompt | llm | StrOutputParser`.
4. **Escalate** — `escalate=true` when `confidence < ESCALATE_THRESHOLD` (0.5).

Prior turns of a session feed back into both chains via a `{history}` variable.

## API

`GET /health` → `{"status":"ok","approach":"langchain","usecase":"05-support-triage"}`

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
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration
```

Copy `.env.example` to `.env` and set `LLM_GATEWAY_KEY`. Default model is the free
local alias `qwen-local-instruct`. Optional `LLM_TEMPERATURE` (default `0.0`) and
`LLM_MAX_TOKENS` (default `384`) tune generation.

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.
