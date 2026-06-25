# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langgraph). See langgraph/10-hitl-approval/README.md
"""FastAPI app for UC10 hitl-approval (langgraph approach) — the showcase.

``POST /run`` invokes the graph under a fresh ``thread_id == run_id``; the graph
runs ``draft`` then SUSPENDS at ``interrupt()`` in ``review``. We read the
interrupt payload and return ``status=awaiting_approval`` plus the proposed
action. ``POST /resume`` re-enters the SAME thread with ``Command(resume=...)``;
the checkpointer restores state and the graph continues into ``execute``.

The LLM and the compiled graph live on ``app.state``; unit tests inject a fake
LLM so nothing hits the network. Because a checkpointer persists per-thread
state, a single compiled graph serves every run_id.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command
from pydantic import BaseModel, Field

from .hitl import build_approval_graph
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "10-hitl-approval"


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


def _config(run_id: str) -> dict:
    # thread_id == run_id: this is the key the checkpointer saves/restores under.
    return {"configurable": {"thread_id": run_id}}


def _interrupt_payload(graph, config: dict) -> dict:
    """Read the pending interrupt's value from the paused checkpoint state.

    When the graph is suspended at ``interrupt()``, the payload lives on the
    paused task's ``interrupts``. (langgraph 0.2 does not echo ``__interrupt__``
    on ``invoke``'s return, so we read it from the durable state instead.)
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for itr in getattr(task, "interrupts", ()):  # Interrupt(value=...)
            if isinstance(itr.value, dict):
                return itr.value
    return {}


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "graph", None) is None:
            app.state.llm = llm or build_llm(settings)
            app.state.graph = build_approval_graph(app.state.llm, settings)
        yield

    app = FastAPI(title="langgraph 10-hitl-approval", lifespan=lifespan)

    # Build eagerly when injected (tests use TestClient without lifespan context).
    if llm is not None:
        app.state.llm = llm
        app.state.graph = build_approval_graph(llm, settings)
    else:
        app.state.graph = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        run_id = uuid.uuid4().hex
        config = _config(run_id)
        # Run until the interrupt in `review`; the graph SUSPENDS there.
        app.state.graph.invoke(
            {
                "request": req.request,
                "proposed_action": "",
                "approved": None,
                "feedback": None,
                "status": "",
                "result": None,
            },
            config=config,
        )
        # The interrupt payload is surfaced on the paused state's pending task.
        proposed = _interrupt_payload(app.state.graph, config).get(
            "proposed_action", ""
        )
        return RunResponse(
            run_id=run_id, status="awaiting_approval", proposed_action=proposed
        )

    @app.post("/resume", response_model=ResumeResponse)
    def resume(req: ResumeRequest) -> ResumeResponse:
        config = _config(req.run_id)
        # No saved checkpoint for this thread => unknown/terminal run => 404.
        snapshot = app.state.graph.get_state(config)
        if not snapshot.next:
            raise HTTPException(status_code=404, detail="unknown run_id")

        out = app.state.graph.invoke(
            Command(resume={"approved": req.approved, "feedback": req.feedback}),
            config=config,
        )
        return ResumeResponse(
            status=out["status"],
            result=out.get("result"),
            feedback=out.get("feedback"),
        )

    return app


app = create_app()
