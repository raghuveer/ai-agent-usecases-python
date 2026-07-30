# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langgraph). See langgraph/07-multi-agent/README.md
"""FastAPI app for UC7 multi-agent (langgraph approach).

``create_app`` accepts an injected LLM so unit tests run offline. ``POST /run``
invokes the multi-agent StateGraph (researcher → writer → reviewer →
(revise loop | END)).
"""
from __future__ import annotations

import json
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .graph import iter_multi_agent, run_multi_agent
from .llm import build_llm
from . import trace as trace_mod
from .settings import get_settings

APPROACH = "langgraph"
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
            "draft": result.get("draft", ""),
            "review": result.get("review", ""),
            "approved": result.get("approved", False),
            "revisions": result.get("revisions", 0),
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "llm", None) is None:
            app.state.llm = llm or build_llm(settings)
        yield

    app = FastAPI(title="langgraph 07-multi-agent", lifespan=lifespan)
    # Inject eagerly when given (tests use TestClient without lifespan context).
    app.state.llm = llm

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Server-sent events: the graph reports its own hand-offs.

        `role` frames here come from `stream_mode="updates"` — the framework
        knows which node ran, so unlike `raw-api/07` nothing in this file has to
        narrate the orchestration. See docs/streaming.md.
        """

        def frames():
            try:
                for event in iter_multi_agent(
                    req.topic,
                    llm=app.state.llm,
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
        """Run the team. ``?trace=1`` records the calls and the graph route."""
        llm = app.state.llm
        recording = trace or settings.trace_sink != "none"
        tracer = None
        callbacks = None
        if recording:
            tracer = trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                include_prompts=settings.trace_include_prompts,
            )
            # In the graph's run config, not on the model — see graph.py.
            callbacks = [trace_mod.TracingCallbackHandler(tracer)]

        result = run_multi_agent(
            req.topic,
            llm=llm,
            research_top_k=settings.research_top_k,
            max_revisions=settings.max_revisions,
            callbacks=callbacks,
        )

        doc = None
        if tracer is not None:
            doc = tracer.finish(
                status="ok" if result["approved"] else "capped",
                stop_reason="approved" if result["approved"] else "max_revisions",
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            draft=result["draft"],
            review=result["review"],
            approved=result["approved"],
            contributions=Contributions(**result["contributions"]),
            trace=doc if trace else None,
        )

    return app


app = create_app()
