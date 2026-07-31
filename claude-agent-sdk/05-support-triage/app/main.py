# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC05 Customer support triage (claude-agent-sdk). See claude-agent-sdk/05-support-triage/README.md
"""FastAPI app for UC05 support-triage (claude-agent-sdk approach)."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .settings import get_settings
from .triage import ORDERS, TRIAGE_SCHEMA, triage

APPROACH = "claude-agent-sdk"
USECASE = "05-support-triage"


class RunRequest(BaseModel):
    ticket: str = Field(max_length=8000)


class RunResponse(BaseModel):
    valid: bool
    decision: dict[str, Any] | None
    errors: list[str]
    order_lookups: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 05-support-triage")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/schema")
    def schema() -> dict:
        """The triage contract the agent must satisfy."""
        return TRIAGE_SCHEMA

    @app.get("/orders")
    def orders() -> dict:
        """The stand-in order system the lookup tool reads."""
        return ORDERS

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await triage(req.ticket, settings, app.state.runner)
        return RunResponse(
            valid=out.valid,
            decision=out.decision,
            errors=out.errors,
            order_lookups=out.order_lookups,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            stop_reason=out.stop_reason,
        )

    return app


app = create_app()
