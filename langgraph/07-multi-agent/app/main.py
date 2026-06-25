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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .graph import run_multi_agent
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
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

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        result = run_multi_agent(
            req.topic,
            llm=app.state.llm,
            research_top_k=settings.research_top_k,
            max_revisions=settings.max_revisions,
        )
        return RunResponse(
            draft=result["draft"],
            review=result["review"],
            approved=result["approved"],
            contributions=Contributions(**result["contributions"]),
        )

    return app


app = create_app()
