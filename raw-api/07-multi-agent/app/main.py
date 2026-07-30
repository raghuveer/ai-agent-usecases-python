# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""FastAPI app for UC7 multi-agent, raw-api approach.

Wires an injectable LLM client onto ``app.state``; unit tests override it and
run fully offline. ``POST /run`` drives the hand-coded orchestrator
(researcher → writer → reviewer, with one revise loop).
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import agents, llm, trace as trace_mod
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "07-multi-agent"


class RunRequest(BaseModel):
    topic: str = Field(max_length=8000)


class Contributions(BaseModel):
    research: str
    writer: str
    reviewer: str


class RunResponse(BaseModel):
    draft: str
    review: str
    approved: bool
    contributions: Contributions
    # Present only with `?trace=1`. Schema: docs/trace-format.md
    trace: dict[str, Any] | None = None


def _sse(event: dict) -> str:
    """Format one orchestration event as a server-sent event frame."""
    payload = dict(event)
    name = payload.pop("type")
    if "result" in payload:
        result = payload.pop("result")
        payload = {
            "draft": result.draft,
            "review": result.review,
            "approved": result.approved,
            "revisions": result.revisions,
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 07-multi-agent")
    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _llm_call(
        system: str, user: str, tracer: trace_mod.Tracer | None = None
    ) -> str:
        settings = app.state.settings
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system=system,
            user=user,
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

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Server-sent events: watch the roles hand off in real time.

        `role` frames are what this use case adds over UC08 — you can see the
        orchestration order, and that with raw-api it is enforced by
        ``iter_orchestrate`` and nothing else. See docs/streaming.md.
        """
        settings = app.state.settings

        def frames():
            def stream_call(system: str, user: str):
                return llm.chat_stream(
                    app.state.client,
                    model=settings.llm_model,
                    system=system,
                    user=user,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )

            try:
                for event in agents.iter_orchestrate(
                    req.topic,
                    llm_stream=stream_call,
                    research_top_k=settings.research_top_k,
                    max_revisions=settings.max_revisions,
                ):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - a silent stream looks finished
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Run the team. ``?trace=1`` returns every call the roles made."""
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

        result = agents.orchestrate(
            req.topic,
            llm_call=(lambda s, u: _llm_call(s, u, tracer)),
            research_top_k=settings.research_top_k,
            max_revisions=settings.max_revisions,
        )

        doc = None
        if tracer is not None:
            doc = tracer.finish(
                status="ok" if result.approved else "capped",
                stop_reason="approved" if result.approved else "max_revisions",
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            draft=result.draft,
            review=result.review,
            approved=result.approved,
            contributions=Contributions(**result.contributions),
            trace=doc if trace else None,
        )

    return app


app = create_app()
