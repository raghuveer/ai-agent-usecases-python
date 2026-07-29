# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (claude-agent-sdk). See claude-agent-sdk/10-hitl-approval/README.md
"""Unit tests for UC10 hitl-approval (claude-agent-sdk). Stubbed agent, no network.

The stub runner deliberately *calls the real gate* (``options.can_use_tool``)
rather than faking approval, so these tests exercise the actual park/resume
handshake — the whole point of this use case — without spawning the CLI.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from fastapi.testclient import TestClient

from app.agent import AgentResult
from app.approval import GUARDED_TOOL, PendingRun, make_gate
from app.main import create_app

DRAFT = "To: customer@example.com\nYour refund of $40 has been approved."


def make_runner(*, call_guarded: bool = True):
    """Stub agent: optionally exercises the guarded tool through the real gate."""

    async def runner(prompt, options) -> AgentResult:
        if not call_guarded:
            return AgentResult(text="No action was necessary.", num_turns=1)

        decision = await options.can_use_tool(
            GUARDED_TOOL,
            {"to": "customer@example.com", "body": "Your refund of $40 has been approved."},
            ToolPermissionContext(),
        )
        if isinstance(decision, PermissionResultDeny):
            return AgentResult(text="", num_turns=2, stop_reason="denied")
        return AgentResult(text="Message delivered to customer@example.com.", num_turns=2)

    return runner


def make_app(**kw):
    return create_app(runner=make_runner(**kw))


# NOTE: every test that parks a run uses `with TestClient(...)`. Un-context-managed,
# TestClient runs each request in its own short-lived event loop, which would kill
# the parked agent task the moment /run returns. Context-managing it keeps one loop
# alive for the client's lifetime — which is what a real single-process uvicorn
# server does, and what this design requires (see README: one worker, one loop).


def test_health():
    with TestClient(make_app()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "claude-agent-sdk",
        "usecase": "10-hitl-approval",
    }


def test_run_parks_and_surfaces_proposed_action():
    with TestClient(make_app()) as client:
        r = client.post("/run", json={"request": "approve a $40 refund"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "awaiting_approval"
        assert "refund of $40" in body["proposed_action"]
        assert body["run_id"]


def test_run_does_not_execute_before_resume():
    """The guarded tool must not have run yet — the agent is parked in the gate."""
    app = make_app()
    with TestClient(app) as client:
        run_id = client.post("/run", json={"request": "refund"}).json()["run_id"]
        pending = app.state.registry.get(run_id)
        assert pending is not None
        assert pending.awaiting_approval is True
        assert pending.task is not None and not pending.task.done()


def test_resume_approved_executes():
    with TestClient(make_app()) as client:
        run_id = client.post("/run", json={"request": "refund"}).json()["run_id"]
        r = client.post("/resume", json={"run_id": run_id, "approved": True})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "executed"
        assert "delivered" in body["result"]


def test_resume_rejected_does_not_execute():
    with TestClient(make_app()) as client:
        run_id = client.post("/run", json={"request": "refund"}).json()["run_id"]
        r = client.post(
            "/resume",
            json={"run_id": run_id, "approved": False, "feedback": "too much"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected"
        assert body["result"] is None
        assert body["feedback"] == "too much"


def test_resume_unknown_run_id_404():
    with TestClient(make_app()) as client:
        r = client.post("/resume", json={"run_id": "nope", "approved": True})
    assert r.status_code == 404


def test_resume_twice_is_404():
    with TestClient(make_app()) as client:
        run_id = client.post("/run", json={"request": "refund"}).json()["run_id"]
        first = client.post("/resume", json={"run_id": run_id, "approved": True})
        assert first.status_code == 200
        second = client.post("/resume", json={"run_id": run_id, "approved": True})
        assert second.status_code == 404


def test_agent_that_never_calls_guarded_tool_completes_immediately():
    with TestClient(make_app(call_guarded=False)) as client:
        body = client.post("/run", json={"request": "just say hi"}).json()
    assert body["status"] == "completed_without_approval"
    assert body["proposed_action"] == "No action was necessary."


@pytest.mark.anyio
async def test_gate_allows_unguarded_tools_without_parking():
    """Only the high-risk tool is gated; Read/Grep etc. pass straight through."""
    pending = PendingRun(run_id="r1")
    gate = make_gate(pending)
    decision = await gate("Read", {"file_path": "a.txt"}, ToolPermissionContext())
    assert isinstance(decision, PermissionResultAllow)
    assert pending.requested.is_set() is False


@pytest.mark.anyio
async def test_default_runner_uses_streaming_input_when_gated(monkeypatch):
    """Regression: `can_use_tool` only works in streaming mode.

    The SDK raises `ValueError: can_use_tool callback requires streaming mode`
    if given a bare string prompt while a permission callback is set. The stub
    runner used everywhere else never calls `query()`, so only this test (and a
    live run) catches it. Found by the live integration test, fixed in
    `default_runner`.
    """
    from collections.abc import AsyncIterable

    import app.agent as agent_mod

    seen: dict = {}

    def fake_query(*, prompt, options):
        seen["prompt"] = prompt

        async def _empty():
            return
            yield  # pragma: no cover

        return _empty()

    monkeypatch.setattr(agent_mod, "query", fake_query)

    pending = PendingRun(run_id="r3")
    gated = ClaudeAgentOptions(can_use_tool=make_gate(pending))
    await agent_mod.default_runner("do the thing", gated)
    assert isinstance(seen["prompt"], AsyncIterable), (
        "a gated run must stream its prompt, not pass a string"
    )

    # Ungated runs keep the simpler string path.
    await agent_mod.default_runner("do the thing", ClaudeAgentOptions())
    assert isinstance(seen["prompt"], str)


@pytest.mark.anyio
async def test_stream_prompt_emits_the_shape_the_cli_expects():
    from app.agent import stream_prompt

    msgs = [m async for m in stream_prompt("hello")]
    assert msgs == [
        {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": "hello"},
            "parent_tool_use_id": None,
        }
    ]


@pytest.mark.anyio
async def test_gate_denies_with_feedback_and_interrupts():
    pending = PendingRun(run_id="r2")
    gate = make_gate(pending)
    task = asyncio.create_task(
        gate(GUARDED_TOOL, {"to": "x", "body": "y"}, ToolPermissionContext())
    )
    await pending.requested.wait()
    pending.feedback = "not authorised"
    pending.decision.set_result(False)

    decision = await task
    assert isinstance(decision, PermissionResultDeny)
    assert decision.message == "not authorised"
    assert decision.interrupt is True


@pytest.mark.anyio
async def test_denial_interrupt_error_is_translated_to_rejection():
    """A denied run stops via interrupt=True, which the SDK raises as an error.

    That is the expected terminal state here, so resolve_run translates it.
    Regression for a bug the live integration test caught.
    """
    from app.approval import resolve_run

    pending = PendingRun(run_id="r4")

    async def boom() -> AgentResult:
        raise Exception("Claude Code returned an error result: [ede_diagnostic]")

    pending.task = asyncio.create_task(boom())
    out = await resolve_run(pending, approved=False, feedback="no")
    assert out.stop_reason == "denied"
    assert out.text == ""


@pytest.mark.anyio
async def test_errors_on_the_approved_path_still_propagate():
    """Only denials are translated — a genuine failure must not look like success."""
    from app.approval import resolve_run

    pending = PendingRun(run_id="r5")

    async def boom() -> AgentResult:
        raise Exception("genuine transport failure")

    pending.task = asyncio.create_task(boom())
    with pytest.raises(Exception, match="genuine transport failure"):
        await resolve_run(pending, approved=True)
