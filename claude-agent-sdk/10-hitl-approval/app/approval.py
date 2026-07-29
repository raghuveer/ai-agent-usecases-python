# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (claude-agent-sdk). See claude-agent-sdk/10-hitl-approval/README.md
"""Human-in-the-loop approval via the Agent SDK's native permission callback.

This is the showcase for this approach. The SDK already has a first-class place
to put a human in the loop: ``ClaudeAgentOptions.can_use_tool``. It is an async
callback invoked **before** any tool runs, and whatever it returns decides the
tool's fate:

* :class:`PermissionResultAllow` — run the tool (optionally with rewritten input)
* :class:`PermissionResultDeny` — refuse it, and tell the model why

Because the callback is ``async``, "ask a human" is just *awaiting a future*. The
agent loop parks itself mid-run at exactly the right point, holding its own state
in the live coroutine — there is no state machine to persist and rehydrate.

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

Contrast with the sibling implementations: ``raw-api/10`` hand-builds a
``CheckpointStore`` plus a resume entry point that re-threads state by hand, and
``langgraph/10`` uses ``interrupt()`` + a checkpointer. Here the "checkpoint" is
the suspended coroutine itself.

**Trade-off, stated honestly:** that is also this design's limitation. A parked
coroutine lives in one process's memory, so a restart loses in-flight approvals
and it does not scale across workers. LangGraph's checkpointer persists the
paused state and survives both. For a durable queue you would still reach for
``langgraph/10`` — see this project's README.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    create_sdk_mcp_server,
    tool,
)

from .agent import AgentResult, Runner, default_runner

# The one high-risk action. Custom SDK tools are namespaced
# ``mcp__<server>__<tool>``, which is the name the permission callback sees.
GUARDED_TOOL = "mcp__approval__send_customer_message"


@tool(
    "send_customer_message",
    "Send a message to a customer. HIGH RISK: requires human approval.",
    {"to": str, "body": str},
)
async def send_customer_message(args: dict[str, Any]) -> dict[str, Any]:
    """Executes only after a human approves — the gate runs before this body."""
    return {
        "content": [
            {"type": "text", "text": f"Message delivered to {args['to']}."}
        ]
    }


def build_approval_server():
    return create_sdk_mcp_server(
        name="approval", version="1.0.0", tools=[send_customer_message]
    )


@dataclass
class PendingRun:
    """One in-flight run and its approval handshake."""

    run_id: str
    requested: asyncio.Event = field(default_factory=asyncio.Event)
    decision: asyncio.Future[bool] = field(default_factory=asyncio.Future)
    proposed_tool: str | None = None
    proposed_input: dict[str, Any] | None = None
    feedback: str | None = None
    task: asyncio.Task[AgentResult] | None = None

    @property
    def awaiting_approval(self) -> bool:
        return self.requested.is_set() and not self.decision.done()

    def proposed_action(self) -> str:
        """Human-readable rendering of the parked tool call."""
        if not self.proposed_input:
            return ""
        to = self.proposed_input.get("to", "")
        body = self.proposed_input.get("body", "")
        return f"To: {to}\n{body}".strip()


class ApprovalRegistry:
    """In-process registry of parked runs, keyed by run_id.

    In-process is the correct scope: the thing being tracked is a suspended
    coroutine, which cannot outlive this process anyway.
    """

    def __init__(self) -> None:
        self._runs: dict[str, PendingRun] = {}

    def create(self, run_id: str) -> PendingRun:
        pending = PendingRun(run_id=run_id)
        self._runs[run_id] = pending
        return pending

    def get(self, run_id: str) -> PendingRun | None:
        return self._runs.get(run_id)

    def discard(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


def make_gate(pending: PendingRun, guarded_tool: str = GUARDED_TOOL):
    """Build the ``can_use_tool`` callback for one run.

    Anything that is not the guarded tool is allowed straight through, so the
    agent can freely read/think; only the irreversible action is gated.
    """

    async def can_use_tool(
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        if tool_name != guarded_tool:
            return PermissionResultAllow(updated_input=input_data)

        pending.proposed_tool = tool_name
        pending.proposed_input = dict(input_data)
        pending.requested.set()  # unblocks /run so it can answer the caller

        approved = await pending.decision  # ⏸ parked until /resume

        if approved:
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(
            message=pending.feedback or "Rejected by the human reviewer.",
            interrupt=True,  # stop the run rather than let it retry another way
        )

    return can_use_tool


async def start_run(
    pending: PendingRun,
    prompt: str,
    options: ClaudeAgentOptions,
    runner: Runner | None = None,
) -> AgentResult | None:
    """Start the agent and return once it either parks for approval or finishes.

    Returns ``None`` when the run parked (caller should report
    ``awaiting_approval``), or the finished :class:`AgentResult` when the agent
    completed without ever touching the guarded tool.
    """
    runner = runner or default_runner
    pending.task = asyncio.create_task(runner(prompt, options))
    waiter = asyncio.create_task(pending.requested.wait())

    done, _ = await asyncio.wait(
        {pending.task, waiter}, return_when=asyncio.FIRST_COMPLETED
    )
    waiter.cancel()

    if pending.task in done:
        # Finished without requesting approval (or raised — surface that).
        return pending.task.result()
    return None


async def resolve_run(
    pending: PendingRun, approved: bool, feedback: str | None = None
) -> AgentResult:
    """Deliver the human decision and wait for the agent to finish.

    A denial uses ``interrupt=True``, which stops the run rather than letting the
    model try a different route to the same action. The SDK surfaces that abrupt
    stop as an **error result**, and ``query()`` raises. For this use case that
    is the *expected terminal state*, not a failure — so it is translated into a
    normal rejected :class:`AgentResult` instead of propagating a 500. Errors on
    the approved path still propagate, because there they are genuine.
    """
    pending.feedback = feedback
    if not pending.decision.done():
        pending.decision.set_result(approved)
    assert pending.task is not None, "resolve_run called before start_run"
    try:
        return await pending.task
    except Exception:
        if approved:
            raise
        return AgentResult(text="", is_error=False, stop_reason="denied")
