# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langchain). See langchain/10-hitl-approval/README.md
"""FastAPI app for UC10 hitl-approval (langchain approach).

``POST /run`` runs the draft chain to completion, then manually pauses: it stashes
the run in a ``RunRegistry`` and returns ``status=awaiting_approval``. ``POST
/resume`` looks the run up by ``run_id`` and continues (approved -> executed,
not approved -> rejected, unknown run_id -> 404).

The LLM/chain and the registry live on ``app.state``; unit tests inject a fake
LLM so nothing hits the network.
"""
from __future__ import annotations

import json
from typing import Any

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from . import hitl
from .llm import build_llm
from . import trace as trace_mod
from .settings import get_settings

APPROACH = "langchain"
USECASE = "10-hitl-approval"


class RunRequest(BaseModel):
    request: str = Field(max_length=8000)


class RunResponse(BaseModel):
    run_id: str
    status: str
    proposed_action: str
    # Present only with `?trace=1`. Schema: docs/trace-format.md
    trace: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    run_id: str = Field(max_length=200)
    approved: bool
    feedback: str | None = Field(default=None, max_length=8000)


class ResumeResponse(BaseModel):
    status: str
    result: str | None = None
    feedback: str | None = None


def _sse(event: dict) -> str:
    """Format one HITL event as a server-sent event frame.

    The `awaiting_approval` frame is the last one a run emits before the human
    decides — see docs/streaming.md for why the stream ends there.
    """
    payload = dict(event)
    name = payload.pop("type")
    if "result" in payload and isinstance(payload["result"], dict):
        payload = payload.pop("result")
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "chain", None) is None:
            app.state.llm = llm or build_llm(settings)
            app.state.chain = hitl.build_draft_chain(app.state.llm, settings)
        yield

    app = FastAPI(title="langchain 10-hitl-approval", lifespan=lifespan)

    # Build eagerly when injected (tests use TestClient without lifespan context).
    if llm is not None:
        app.state.llm = llm
        app.state.chain = hitl.build_draft_chain(llm, settings)
    else:
        app.state.chain = None
    app.state.registry = hitl.RunRegistry()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Stream the draft, then **end at the approval gate**.

        Same contract as `raw-api/10`; the difference is that the tokens come
        from `chain.stream()` rather than hand-threaded deltas. See
        docs/streaming.md.
        """

        def frames():
            try:
                for event in hitl.iter_start_run(
                    req.request,
                    chain=app.state.chain,
                    registry=app.state.registry,
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
    def resume_stream(req: ResumeRequest) -> StreamingResponse:
        """The second stream, opened after the human decides."""

        def frames():
            try:
                for event in hitl.iter_resume_run(
                    req.run_id,
                    approved=req.approved,
                    feedback=req.feedback,
                    registry=app.state.registry,
                ):
                    yield _sse(event)
            except hitl.UnknownRunError:
                yield _sse({"type": "error", "message": "unknown run_id"})
            except Exception as exc:  # noqa: BLE001
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Draft and park. ``?trace=1`` records the drafting call only.

        The trace ends where the run does — at the gate.
        """
        chain = app.state.chain
        recording = trace or settings.trace_sink != "none"
        tracer = None
        if recording:
            tracer = trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                include_prompts=settings.trace_include_prompts,
            )
            chain = chain.with_config(
                callbacks=[trace_mod.TracingCallbackHandler(tracer)]
            )

        paused = hitl.start_run(
            req.request, chain=chain, registry=app.state.registry
        )

        doc = None
        if tracer is not None:
            doc = tracer.finish(status="ok", stop_reason="awaiting_approval")
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            run_id=paused.run_id,
            status=paused.status,
            proposed_action=paused.proposed_action,
            trace=doc if trace else None,
        )

    @app.post("/resume", response_model=ResumeResponse)
    def resume(req: ResumeRequest) -> ResumeResponse:
        try:
            out = hitl.resume_run(
                req.run_id,
                approved=req.approved,
                feedback=req.feedback,
                registry=app.state.registry,
            )
        except hitl.UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return ResumeResponse(**out)

    return app


app = create_app()
