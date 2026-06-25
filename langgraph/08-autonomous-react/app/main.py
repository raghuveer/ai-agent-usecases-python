# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langgraph). See langgraph/08-autonomous-react/README.md
"""FastAPI app for UC8 autonomous-react (langgraph approach).

``create_app`` accepts an injected LLM so unit tests run offline. ``POST /run``
invokes the ReAct StateGraph (reason → act → observe → loop | END).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .llm import build_llm
from .react import run_react
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "08-autonomous-react"


class RunRequest(BaseModel):
    task: str = Field(max_length=8000)
    max_steps: int | None = Field(default=None, ge=1, le=12)


class StepModel(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class RunResponse(BaseModel):
    answer: str
    steps: list[StepModel]
    stopped_reason: str


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "llm", None) is None:
            app.state.llm = llm or build_llm(settings)
        yield

    app = FastAPI(title="langgraph 08-autonomous-react", lifespan=lifespan)
    # Inject eagerly when given (tests use TestClient without lifespan context).
    app.state.llm = llm

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
        result = run_react(req.task, llm=app.state.llm, max_steps=max_steps)
        return RunResponse(
            answer=result["answer"],
            steps=[StepModel(**s) for s in result["steps"]],
            stopped_reason=result["stopped_reason"],
        )

    return app


app = create_app()
