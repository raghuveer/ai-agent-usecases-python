# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langgraph). See langgraph/05-support-triage/README.md
"""FastAPI app for UC5 support-triage (langgraph approach).

On startup it builds the LLM client and compiles the triage graph.
``create_app`` accepts an injected LLM so unit tests run offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from . import triage
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "05-support-triage"


class RunRequest(BaseModel):
    message: str
    session_id: str | None = None


class RunResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    escalate: bool


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "graph"):
            app.state.llm = llm or build_llm(settings)
            app.state.graph = triage.build_triage_graph(app.state.llm, settings)
        yield

    app = FastAPI(title="langgraph 05-support-triage", lifespan=lifespan)

    # When the LLM is injected (tests), compile eagerly so requests work even if
    # the lifespan startup hasn't run (e.g. TestClient used without a context).
    if llm is not None:
        app.state.llm = llm
        app.state.graph = triage.build_triage_graph(llm, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        result = triage.run_triage(
            app.state.graph, req.message, session_id=req.session_id
        )
        return RunResponse(**result)

    return app


app = create_app()
