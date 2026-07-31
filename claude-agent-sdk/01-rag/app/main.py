# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC01 Q&A / RAG (claude-agent-sdk). See claude-agent-sdk/01-rag/README.md
"""FastAPI app for UC01 rag (claude-agent-sdk approach)."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .rag import answer
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "01-rag"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    answer: str
    sources: list[str]
    searches: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 01-rag")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await answer(req.question, settings, app.state.runner)
        return RunResponse(
            answer=out.answer,
            sources=out.sources,
            searches=out.searches,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            stop_reason=out.stop_reason,
        )

    return app


app = create_app()
