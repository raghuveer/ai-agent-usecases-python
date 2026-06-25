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

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import hitl, llm
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "10-hitl-approval"


class RunRequest(BaseModel):
    request: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    proposed_action: str


class ResumeRequest(BaseModel):
    run_id: str
    approved: bool
    feedback: str | None = None


class ResumeResponse(BaseModel):
    status: str
    result: str | None = None
    feedback: str | None = None


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

    def _llm_call(user_prompt: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=hitl.DRAFT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        paused = hitl.start_run(
            req.request, llm_call=_llm_call, store=app.state.store
        )
        return RunResponse(
            run_id=paused.run_id,
            status=paused.status,
            proposed_action=paused.proposed_action,
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
