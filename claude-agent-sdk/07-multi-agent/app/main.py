# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""FastAPI app for UC07 multi-agent (claude-agent-sdk approach) — the showcase."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .settings import get_settings
from .team import TEAM, run_team

APPROACH = "claude-agent-sdk"
USECASE = "07-multi-agent"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    report: str
    subagents_used: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 07-multi-agent")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/team")
    def team() -> dict:
        """Expose the roster so the least-privilege split is inspectable."""
        return {
            name: {"description": a.description, "tools": a.tools}
            for name, a in TEAM.items()
        }

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await run_team(req.question, settings, app.state.runner)
        return RunResponse(
            report=out.report,
            subagents_used=out.subagents_used,
            tools_used=out.tools_used,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
        )

    return app


app = create_app()
