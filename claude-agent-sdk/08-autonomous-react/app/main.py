# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC08 Autonomous ReAct (claude-agent-sdk). See claude-agent-sdk/08-autonomous-react/README.md
"""FastAPI app for UC08 autonomous-react (claude-agent-sdk approach) — the showcase."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .react_agent import METRICS, run_react
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "08-autonomous-react"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class StepModel(BaseModel):
    tool: str
    input: dict[str, Any]


class RunResponse(BaseModel):
    answer: str
    trace: list[StepModel]
    num_turns: int
    cost_usd: float
    hit_turn_limit: bool


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 08-autonomous-react")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/metrics")
    def metrics() -> dict:
        """The fixed warehouse the agent must query through tools."""
        return METRICS

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await run_react(req.question, settings, app.state.runner)
        return RunResponse(
            answer=out.answer,
            trace=[StepModel(tool=s.tool, input=s.input) for s in out.trace],
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            hit_turn_limit=out.hit_turn_limit,
        )

    return app


app = create_app()
