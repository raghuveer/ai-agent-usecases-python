# UC10 — hitl-approval (claude-agent-sdk) — showcase

A workflow that **pauses for human approval** before executing a high-risk
action. This is a showcase for the Agent SDK: human-in-the-loop is not a pattern
you build here, it is a callback you fill in.

## The mechanism

`ClaudeAgentOptions.can_use_tool` is an **async** callback the SDK invokes before
any tool runs. Return `PermissionResultAllow` and the tool runs; return
`PermissionResultDeny` and it does not, with your message handed back to the
model. Because it is async, "ask a human" is just *awaiting a future*:

```
/run     ──► agent runs ──► calls send_customer_message
                                   │
                             can_use_tool fires
                                   │
                      records proposal, sets `requested`,
                             awaits `decision`  ⏸  (agent parked)
                                   │
/resume  ──► decision.set_result(True/False) ──► Allow / Deny
                                   │
                             agent continues ──► result
```

The agent parks itself mid-run holding its own state in the live coroutine. There
is no state machine to serialise and rehydrate.

Only `send_customer_message` — a custom SDK tool, so the callback sees it as
`mcp__approval__send_customer_message` — is gated. Everything else passes
straight through, so the agent can read and reason freely; only the irreversible
step stops.

> ⚠️ **The gate is easy to disable by accident.** Listing the guarded tool in
> `allowed_tools` **auto-approves it before `can_use_tool` is consulted** — the
> agent sends the message and no human ever sees it. This was a real bug here,
> caught only by the live test; the SDK warns with `CanUseToolShadowedWarning`.
> `allowed_tools` is therefore **deliberately empty** in `app/main.py`. The tool
> is still available — it comes from the MCP server, and `allowed_tools` controls
> auto-approval, not availability. Allow-rules in settings files can shadow it
> the same way, which is one more reason `setting_sources=[]` is set.
>
> Two related live findings: a permission callback **requires streaming input**
> (a string prompt raises `ValueError`), and a denial with `interrupt=True`
> surfaces as an *error result* — which `resolve_run` translates into a normal
> rejection, since that is the expected terminal state here.

## How the four approaches compare

| Need | raw-api | langchain | langgraph | claude-agent-sdk |
|---|---|---|---|---|
| Pause mid-run | return early + remember | callback workaround | `interrupt()` | `await` inside `can_use_tool` |
| Persist paused state | hand-built `CheckpointStore` | — | checkpointer (`MemorySaver`) | the suspended coroutine itself |
| Resume | separate entry point re-threads state | — | `Command(resume=...)` | resolve the future |
| Gate a *specific* action | manual branch | manual branch | route to an interrupt node | tool name check in the callback |

## Where this design gives way to langgraph

Being the suspended coroutine is what makes this so small, and it is also the
limitation:

- **Single process, single worker.** A parked run lives in one event loop's
  memory. Run this app with more than one uvicorn worker and `/resume` may land
  on a process that never saw the `/run`.
- **No durability.** Restart the process and in-flight approvals are gone.
  `langgraph/10`'s checkpointer survives both — swap in `SqliteSaver`/
  `PostgresSaver` there and paused runs outlive the process.
- ~~**No TTL/eviction.**~~ **Fixed.** A run nobody resolves is auto-**denied**
  and reaped after `APPROVAL_TTL_SECONDS`, and `APPROVAL_MAX_PENDING` caps how
  many can be parked at once (`/run` returns **429** past it). Deny-on-timeout
  is deliberate: silence must never be read as consent for a high-risk action.

If approvals must survive restarts or span replicas, use
[`langgraph/10-hitl-approval`](../../langgraph/10-hitl-approval). If you want the
gate to be three lines inside the agent you already have, use this.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"10-hitl-approval"}`
- `POST /run` body `{"request": str}` →
  `{"run_id": str, "status": "awaiting_approval", "proposed_action": str}`
  (or `status: "completed_without_approval"` if the agent never reached the
  guarded tool)
- `POST /resume` body `{"run_id": str, "approved": bool, "feedback": str|null}` →
  approved → `{"status":"executed","result": ...}`;
  not approved → `{"status":"rejected","result": null}`;
  unknown, already-resumed, or timed-out `run_id` → **404**.

`POST /run` returns **429** when `APPROVAL_MAX_PENDING` runs are already parked.

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — note: no `/v1` suffix (the SDK appends `/v1/messages`) |
| `LLM_GATEWAY_KEY` | placeholder | platform virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | allow-listed alias; `claude-sonnet` for harder runs |
| `AGENT_MAX_TURNS` | `12` | hard cap on agent turns |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard cap on spend per run |
| `AGENT_EFFORT` | `low` | thinking depth |
| `APPROVAL_TTL_SECONDS` | `900` | unresolved runs are auto-**denied** and reaped |
| `APPROVAL_MAX_PENDING` | `50` | cap on concurrently parked runs; `/run` → 429 past it |

**Auth gotcha:** the gateway requires `Authorization: Bearer`. The SDK sends that
only when `ANTHROPIC_AUTH_TOKEN` is set; `ANTHROPIC_API_KEY` makes it send
`x-api-key`, which the gateway rejects with 401. `app/agent.py` sets the right
one for you.

## Prerequisites

- **Python 3.12+** — all app code here is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH.** The Python SDK spawns the CLI
  as a subprocess; this is inherent to the SDK, not a design choice here.
- Unit tests need **neither** — they inject a stub runner.

## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent (needs the gateway, a key, Node + the CLI; spends a little budget):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve (single worker — see the caveat above):
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"request":"Tell customer@example.com their $40 refund is approved."}'
#   curl -X POST localhost:8000/resume -H 'content-type: application/json' \
#     -d '{"run_id":"<id from /run>","approved":true}'
```
