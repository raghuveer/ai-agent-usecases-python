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

from fastapi import FastAPI
from pydantic import BaseModel

from . import agents, llm
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "07-multi-agent"


class RunRequest(BaseModel):
    topic: str


class Contributions(BaseModel):
    research: str
    writer: str
    reviewer: str


class RunResponse(BaseModel):
    draft: str
    review: str
    approved: bool
    contributions: Contributions


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 07-multi-agent")
    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _llm_call(system: str, user: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system=system,
            user=user,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        result = agents.orchestrate(
            req.topic,
            llm_call=_llm_call,
            research_top_k=settings.research_top_k,
            max_revisions=settings.max_revisions,
        )
        return RunResponse(
            draft=result.draft,
            review=result.review,
            approved=result.approved,
            contributions=Contributions(**result.contributions),
        )

    return app


app = create_app()
