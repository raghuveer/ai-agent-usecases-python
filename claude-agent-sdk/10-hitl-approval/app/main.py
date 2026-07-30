# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (claude-agent-sdk). See claude-agent-sdk/10-hitl-approval/README.md
"""FastAPI app for UC10 hitl-approval (claude-agent-sdk) — the showcase.

``POST /run`` starts an agent that drafts and tries to send a customer message.
The send is a guarded custom tool, so the SDK's ``can_use_tool`` callback fires
first and parks the agent. ``POST /resume`` delivers the human decision, which
un-parks the same live coroutine.
"""
from __future__ import annotations

import json

import uuid

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Runner, build_options
from .approval import (
    iter_resume_run,
    iter_start_run,
    GUARDED_TOOL,
    ApprovalRegistry,
    RegistryFull,
    build_approval_server,
    make_gate,
    resolve_run,
    start_run,
)
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "10-hitl-approval"

SYSTEM_PROMPT = (
    "You are an operations assistant handling a customer request. Draft a short, "
    "professional message (at most 3 sentences) that performs the requested "
    "action, then send it with the send_customer_message tool. A human reviews "
    "every send before it goes out. Do not ask the user for confirmation "
    "yourself — just call the tool."
)


class RunRequest(BaseModel):
    request: str = Field(max_length=8000)


class RunResponse(BaseModel):
    run_id: str
    status: str
    proposed_action: str


class ResumeRequest(BaseModel):
    run_id: str = Field(max_length=200)
    approved: bool
    feedback: str | None = Field(default=None, max_length=8000)


class ResumeResponse(BaseModel):
    status: str
    result: str | None = None
    feedback: str | None = None


def _sse(event: dict) -> str:
    """Format one HITL event as a server-sent event frame. See docs/streaming.md."""
    payload = dict(event)
    name = payload.pop("type")
    if "result" in payload:
        result = payload.pop("result")
        payload = {
            "status": "executed" if payload.pop("approved", True) else "rejected",
            "result": getattr(result, "text", "") or None,
            "num_turns": getattr(result, "num_turns", 0),
            "cost_usd": getattr(result, "cost_usd", 0.0),
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 10-hitl-approval")

    app.state.settings = settings
    app.state.runner = runner
    app.state.registry = ApprovalRegistry(
        ttl_seconds=settings.approval_ttl_seconds,
        max_pending=settings.approval_max_pending,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run/stream")
    async def run_stream(req: RunRequest) -> StreamingResponse:
        """Stream the agent's turns, then **end at the approval gate**.

        The gate here is a ``can_use_tool`` callback that suspends a live
        coroutine, so the agent runs as a background task publishing frames to
        a queue while this drains them. Ending the response leaves the run
        parked — and that is the limitation worth seeing: what survives the
        gate is a coroutine and an in-memory queue, **not a checkpoint**.
        `langgraph/10` resumes from a checkpointer and would survive a restart;
        this would not. See docs/streaming.md.
        """
        run_id = uuid.uuid4().hex
        try:
            pending = app.state.registry.create(run_id)
        except RegistryFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        options = build_options(
            settings,
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=[],
            mcp_servers={"approval": build_approval_server()},
            permission_mode="default",
            can_use_tool=make_gate(pending),
        )

        async def frames():
            try:
                async for event in iter_start_run(
                    pending, req.request, options, app.state.runner
                ):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - a silent stream looks finished
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/resume/stream")
    async def resume_stream(req: ResumeRequest) -> StreamingResponse:
        """Deliver the decision and stream whatever the agent does next."""
        pending = app.state.registry.get(req.run_id)
        if pending is None:
            raise HTTPException(status_code=404, detail="unknown run_id")

        async def frames():
            try:
                async for event in iter_resume_run(
                    pending, req.approved, req.feedback
                ):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001
                yield _sse({"type": "error", "message": str(exc)})
            finally:
                app.state.registry.discard(req.run_id)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        run_id = uuid.uuid4().hex
        try:
            pending = app.state.registry.create(run_id)
        except RegistryFull as exc:
            # Shed load rather than accumulate parked agent coroutines (F12).
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        options = build_options(
            settings,
            system_prompt=SYSTEM_PROMPT,
            # DELIBERATELY EMPTY. Listing GUARDED_TOOL here would auto-approve it
            # *before* can_use_tool is consulted, silently bypassing the gate —
            # the SDK warns `CanUseToolShadowedWarning` when you do. An empty
            # allow-list makes every tool call fall through to the callback,
            # which is the whole point of this use case. The tool is still
            # available: it comes from the MCP server below, and allowed_tools
            # controls auto-approval, not availability.
            allowed_tools=[],
            mcp_servers={"approval": build_approval_server()},
            # The gate replaces blanket permission handling for this run.
            permission_mode="default",
            can_use_tool=make_gate(pending),
        )

        finished = await start_run(pending, req.request, options, app.state.runner)

        if finished is not None:
            # Agent never reached the guarded tool — nothing to approve.
            app.state.registry.discard(run_id)
            return RunResponse(
                run_id=run_id, status="completed_without_approval",
                proposed_action=finished.text,
            )

        return RunResponse(
            run_id=run_id,
            status="awaiting_approval",
            proposed_action=pending.proposed_action(),
        )

    @app.post("/resume", response_model=ResumeResponse)
    async def resume(req: ResumeRequest) -> ResumeResponse:
        pending = app.state.registry.get(req.run_id)
        if pending is None or not pending.awaiting_approval:
            raise HTTPException(status_code=404, detail="unknown run_id")

        result = await resolve_run(pending, req.approved, req.feedback)
        app.state.registry.discard(req.run_id)  # terminal: no second resume

        if req.approved:
            return ResumeResponse(status="executed", result=result.text)
        return ResumeResponse(status="rejected", result=None, feedback=req.feedback)

    return app


app = create_app()
