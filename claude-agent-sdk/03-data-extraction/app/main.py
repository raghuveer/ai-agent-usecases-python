# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC03 Data extraction (claude-agent-sdk). See claude-agent-sdk/03-data-extraction/README.md
"""FastAPI app for UC03 data-extraction (claude-agent-sdk approach)."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .extract import INVOICE_SCHEMA, extract
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "03-data-extraction"


class RunRequest(BaseModel):
    document: str = Field(max_length=40000)


class RunResponse(BaseModel):
    valid: bool
    invoice: dict[str, Any] | None
    errors: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 03-data-extraction")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.get("/schema")
    def schema() -> dict:
        """The tool schema that doubles as the extraction contract."""
        return INVOICE_SCHEMA

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await extract(req.document, settings, app.state.runner)
        return RunResponse(
            valid=out.valid,
            invoice=out.invoice,
            errors=out.errors,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            stop_reason=out.stop_reason,
        )

    return app


app = create_app()
