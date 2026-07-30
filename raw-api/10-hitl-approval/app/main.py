# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (raw-api). See raw-api/10-hitl-approval/README.md
"""FastAPI app for UC10 hitl-approval, raw-api approach.

Two endpoints split by a human checkpoint:
- ``POST /run`` drafts a proposed high-risk action, persists a paused run in the
  hand-built ``CheckpointStore``, and returns ``status=awaiting_approval``.
- ``POST /resume`` looks the run up by ``run_id`` and continues: approved ->
  executed, not approved -> rejected, unknown run_id -> 404.

The LLM client and the checkpoint store live on ``app.state``; unit tests inject
a stub client so nothing hits the network.
"""
from __future__ import annotations

import json
from typing import Any

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import hitl, llm
from . import trace as trace_mod
from .settings import get_settings

APPROACH = "raw-api"
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

def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "client", None) is None:
            app.state.client = llm.build_client(settings)
        yield

    app = FastAPI(title="raw-api 10-hitl-approval", lifespan=lifespan)

    app.state.settings = settings
    # Built lazily in lifespan (or injected by tests before the first request).
    app.state.client = None
    # The hand-built checkpoint store that persists paused runs.
    app.state.store = hitl.CheckpointStore()

    def _llm_call(user_prompt: str, tracer: trace_mod.Tracer | None = None) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=hitl.DRAFT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            on_call=(
                None
                if tracer is None
                else lambda rec: tracer.llm_span(
                    messages=rec["messages"],
                    completion=rec["completion"],
                    duration_ms=rec["duration_ms"],
                    input_tokens=rec["input_tokens"],
                    output_tokens=rec["output_tokens"],
                )
            ),
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Draft and park. ``?trace=1`` records the drafting call.

        The trace covers phase one only — it ends where the run does, at the
        gate. What happens after the human decides is a separate run and a
        separate trace, which is the honest shape for a workflow that pauses.
        """
        settings = app.state.settings
        recording = trace or settings.trace_sink != "none"
        tracer = (
            trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                include_prompts=settings.trace_include_prompts,
            )
            if recording
            else None
        )

        paused = hitl.start_run(
            req.request,
            llm_call=(lambda p: _llm_call(p, tracer)),
            store=app.state.store,
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

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Stream the draft, then **end the stream at the approval gate**.

        The last frame is `awaiting_approval` carrying the run id. The
        connection then closes, on purpose: an approver may take minutes or
        days, and a held-open socket would turn every proxy timeout or restart
        into a lost run. The checkpoint is the contract; the connection is not.
        `POST /resume/stream` opens a fresh stream once the human decides.
        See docs/streaming.md.
        """
        settings = app.state.settings

        def frames():
            def stream_call(user_prompt: str):
                return llm.chat_stream(
                    app.state.client,
                    model=settings.llm_model,
                    system_prompt=hitl.DRAFT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )

            try:
                for event in hitl.iter_start_run(
                    req.request, llm_stream=stream_call, store=app.state.store
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
        """The second stream: the human has decided, so finish the run.

        No `token` frames — approving executes the text that was already
        drafted and reviewed. Regenerating after approval would mean the human
        approved something other than what ships.
        """

        def frames():
            try:
                for event in hitl.iter_resume_run(
                    req.run_id,
                    approved=req.approved,
                    feedback=req.feedback,
                    store=app.state.store,
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

    @app.post("/resume", response_model=ResumeResponse)
    def resume(req: ResumeRequest) -> ResumeResponse:
        try:
            out = hitl.resume_run(
                req.run_id,
                approved=req.approved,
                feedback=req.feedback,
                store=app.state.store,
            )
        except hitl.UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return ResumeResponse(**out)

    return app


app = create_app()
