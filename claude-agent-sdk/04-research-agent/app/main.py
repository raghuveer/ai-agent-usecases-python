# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""FastAPI app for UC04 research-agent (claude-agent-sdk approach)."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .research import research
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "04-research-agent"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    answer: str
    mode: str
    citations: list[str]
    searches: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 04-research-agent")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        # `mode` is surfaced here too so an operator can confirm at a glance
        # whether this instance can reach the network.
        return {
            "status": "ok",
            "approach": APPROACH,
            "usecase": USECASE,
            "mode": "web" if settings.research_allow_web else "offline",
        }

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await research(req.question, settings, app.state.runner)
        return RunResponse(
            answer=out.answer,
            mode=out.mode,
            citations=out.citations,
            searches=out.searches,
            tools_used=out.tools_used,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
        )

    return app


app = create_app()
