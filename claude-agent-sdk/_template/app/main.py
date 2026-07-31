# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — claude-agent-sdk project template. See claude-agent-sdk/_template/README.md
"""Minimal FastAPI skeleton for the claude-agent-sdk approach (template).

`GET /health` reports the approach/usecase. `POST /run` sends the question to a
one-shot agent with no tools and returns its answer. Copy this folder as the
starting point for a new use case.

Routes are `async def` here, unlike the other three approaches: the Agent SDK is
async-first (`query()` is an async generator), so the whole call path is async.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "_template"

SYSTEM_PROMPT = (
    "You are a terse assistant. Answer the user's question in one short "
    "sentence. Do not use tools."
)


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    answer: str
    tools_used: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk template")

    # Injectable runner: unit tests pass a stub so nothing spawns the CLI.
    app.state.settings = settings
    app.state.runner = runner or default_runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        options = build_options(settings, system_prompt=SYSTEM_PROMPT)
        result = await app.state.runner(req.question, options)
        return RunResponse(
            answer=result.text,
            tools_used=result.tool_names,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            stop_reason=outcome_of(result),
        )

    return app


app = create_app()
