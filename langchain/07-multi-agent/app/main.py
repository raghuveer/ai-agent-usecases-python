# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langchain). See langchain/07-multi-agent/README.md
"""FastAPI app for UC7 multi-agent (langchain approach).

Builds an injectable chat model on ``app.state``; unit tests set a
``FakeListChatModel`` and run offline. ``POST /run`` drives the role-chain
orchestrator (researcher → writer → reviewer, with one revise loop).
"""
from __future__ import annotations

import json
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .agents import iter_orchestrate, orchestrate
from .llm import build_llm
from . import trace as trace_mod
from .settings import get_settings

APPROACH = "langchain"
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm()
    yield


app = FastAPI(title="langchain 07-multi-agent", lifespan=lifespan)
# Pre-declare state so tests can inject a fake before the first request.
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run/stream")
def run_stream(req: RunRequest) -> StreamingResponse:
    """Server-sent events: watch the roles hand off in real time.

    LangChain composes streaming through the whole chain
    (``prompt | llm | parser``), so the only change from the blocking path is
    ``chain.stream()`` instead of ``chain.invoke()``. See docs/streaming.md.
    """
    settings = get_settings()

    def frames():
        try:
            for event in iter_orchestrate(
                req.topic,
                llm=app.state.llm,
                research_top_k=settings.research_top_k,
                max_revisions=settings.max_revisions,
                stream=True,
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
    """Run the team. ``?trace=1`` returns every call the roles made.

    As in UC08, the tracer attaches as a LangChain callback — the orchestration
    itself needs no changes. See docs/trace-format.md.
    """
    settings = get_settings()
    llm: BaseChatModel = app.state.llm

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
        llm = llm.with_config(callbacks=[trace_mod.TracingCallbackHandler(tracer)])

    result = orchestrate(
        req.topic,
        llm=llm,
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
