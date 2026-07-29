# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""FastAPI app for UC06 sql-agent (claude-agent-sdk approach)."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import db
from .agent import Runner
from .settings import get_settings
from .sql_agent import ask

APPROACH = "claude-agent-sdk"
USECASE = "06-sql-agent"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    answer: str
    queries: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 06-sql-agent")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/schema")
    def schema() -> dict:
        """The schema the agent discovers for itself, exposed for humans."""
        path = db.ensure_db()
        return {t: db.describe_table(path, t) for t in db.list_tables(path)}

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await ask(req.question, settings, app.state.runner)
        return RunResponse(
            answer=out.answer,
            queries=out.queries,
            tools_used=out.tools_used,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
        )

    return app


app = create_app()
