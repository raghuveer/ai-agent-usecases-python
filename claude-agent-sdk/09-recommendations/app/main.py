# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC09 Personalised recommendations (claude-agent-sdk). See claude-agent-sdk/09-recommendations/README.md
"""FastAPI app for UC09 recommendations (claude-agent-sdk approach)."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .recommend import CATALOG, PROFILES, recommend
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "09-recommendations"


class RunRequest(BaseModel):
    user_id: str = Field(max_length=64)


class RunResponse(BaseModel):
    valid: bool
    items: list[dict[str, Any]]
    rationale: str
    errors: list[str]
    num_turns: int
    cost_usd: float


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 09-recommendations")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/catalog")
    def catalog() -> list[dict]:
        return CATALOG

    @app.get("/profiles")
    def profiles() -> dict:
        return PROFILES

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await recommend(req.user_id, settings, app.state.runner)
        return RunResponse(
            valid=out.valid,
            items=out.items,
            rationale=out.rationale,
            errors=out.errors,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
        )

    return app


app = create_app()
