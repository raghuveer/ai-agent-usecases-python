# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC05 Customer support triage (claude-agent-sdk). See claude-agent-sdk/05-support-triage/README.md
"""Support triage: classify, optionally enrich from the order system, then route.

The routing here is **agentic, not hardcoded**. The other approaches classify
first and then branch in Python — `if category == "shipping": lookup_order(...)`.
Here the agent decides for itself whether an order lookup would change its
answer, and only then calls the tool:

    ticket ──► (maybe) lookup_order ──► emit_triage(category, priority, …)

That flexibility is the trade: a hardcoded branch is predictable and free, while
the agent might skip a lookup you wanted or make one you did not. The decision is
therefore made auditable — the response reports whether a lookup actually
happened.

The final decision arrives through `emit_triage`, whose schema is the contract
(the same tool-as-schema idea as UC03), and is validated with Pydantic before it
leaves this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import BaseModel, Field, ValidationError

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import Settings

Category = Literal["billing", "technical", "shipping", "account", "other"]
Priority = Literal["low", "normal", "high", "urgent"]

# Stand-in order system. Fixed so runs are deterministic and offline.
ORDERS: dict[str, dict[str, Any]] = {
    "A-1001": {"status": "delivered", "carrier": "DHL", "delivered_on": "2026-03-02"},
    "A-1002": {"status": "in_transit", "carrier": "DHL", "eta": "2026-03-19"},
    "A-1003": {"status": "lost_in_transit", "carrier": "Royal Mail", "opened_case": True},
    "A-1004": {"status": "refunded", "refund_amount": 40.0, "refunded_on": "2026-02-27"},
}


class TriageDecision(BaseModel):
    """The contract for a triage outcome."""

    category: Category
    priority: Priority
    needs_human: bool
    reply: str = Field(max_length=2000)


TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "shipping", "account", "other"],
            "description": "Primary topic of the ticket.",
        },
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high", "urgent"],
            "description": "urgent = customer blocked or money at risk.",
        },
        "needs_human": {
            "type": "boolean",
            "description": "True if a human agent must review before replying.",
        },
        "reply": {
            "type": "string",
            "description": "Draft reply to the customer, at most 3 sentences.",
        },
    },
    "required": ["category", "priority", "needs_human", "reply"],
}

LOOKUP_TOOL = "mcp__triage__lookup_order"
EMIT_TOOL = "mcp__triage__emit_triage"


@tool("lookup_order", "Look up an order by its id, e.g. A-1002.", {"order_id": str})
async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
    order_id = str(args.get("order_id", "")).strip().upper()
    order = ORDERS.get(order_id)
    if order is None:
        return {
            "content": [{"type": "text", "text": f"No order found with id {order_id}."}],
            "is_error": True,
        }
    details = ", ".join(f"{k}={v}" for k, v in order.items())
    return {"content": [{"type": "text", "text": f"{order_id}: {details}"}]}


@tool(
    "emit_triage",
    "Record the triage decision. Call this exactly once, last.",
    TRIAGE_SCHEMA,
)
async def emit_triage(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": "Triage recorded."}]}


SYSTEM_PROMPT = """You triage inbound customer support tickets.

EVERY ticket gets a triage decision — including short, simple, or purely
informational questions. You are NOT finished until you have called emit_triage
exactly once. Never reply to the customer in your own message text instead of
calling the tool: the reply belongs in emit_triage's `reply` field.

If the ticket mentions an order id (like A-1002) and the order's status would
change your reply, call lookup_order first. Otherwise do not — do not invent an
order id in order to have something to look up.

Guidance:
- `urgent` means the customer is blocked or money is at risk (lost parcel,
  double charge). Routine questions are `normal` or `low`.
- Set needs_human = true for refunds, chargebacks, complaints about staff, legal
  or data-protection matters, or anything you are unsure about.
- The reply is at most 3 sentences, plain and specific. Never invent order facts
  you did not get from lookup_order."""

TRIAGE_TOOLS = [LOOKUP_TOOL, EMIT_TOOL]


def build_triage_server():
    return create_sdk_mcp_server(
        name="triage", version="1.0.0", tools=[lookup_order, emit_triage]
    )


@dataclass
class TriageResult:
    valid: bool
    decision: dict[str, Any] | None
    errors: list[str]
    order_lookups: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str = "end_turn"


async def triage(
    ticket: str, settings: Settings, runner: Runner | None = None
) -> TriageResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=TRIAGE_TOOLS,
        mcp_servers={"triage": build_triage_server()},
        max_turns=max(settings.agent_max_turns, 5),
    )
    result = await runner(ticket, options)

    lookups = [
        str(c.input.get("order_id", ""))
        for c in result.tool_calls
        if c.name == LOOKUP_TOOL and c.input.get("order_id")
    ]
    emits = [c for c in result.tool_calls if c.name == EMIT_TOOL]

    if not emits:
        return TriageResult(
            valid=False,
            decision=None,
            errors=["agent did not call emit_triage"],
            order_lookups=lookups,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            stop_reason=outcome_of(result),
        )

    try:
        decision = TriageDecision.model_validate(emits[-1].input)
    except ValidationError as exc:
        return TriageResult(
            valid=False,
            decision=None,
            errors=[
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ],
            order_lookups=lookups,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            stop_reason=outcome_of(result),
        )

    return TriageResult(
        valid=True,
        decision=decision.model_dump(),
        errors=[],
        order_lookups=lookups,
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
        stop_reason=outcome_of(result),
    )
