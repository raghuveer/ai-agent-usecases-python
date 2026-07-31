# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC02 Code generation (claude-agent-sdk). See claude-agent-sdk/02-code-generation/README.md
"""FastAPI app for UC02 code-generation (claude-agent-sdk approach)."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Runner
from .codegen import generate
from .settings import get_settings

APPROACH = "claude-agent-sdk"
USECASE = "02-code-generation"


class RunRequest(BaseModel):
    task: str = Field(max_length=8000)


class RunResponse(BaseModel):
    solution: str
    tests: str
    summary: str
    tests_passed: bool
    files: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str
    # False means the shell that just ran was NOT confined (F9). Observed from
    # the CLI's own stderr rather than inferred from config, because the two
    # disagree — see SandboxMonitor. `sandbox_note` carries the CLI's reason.
    sandboxed: bool
    sandbox_note: str | None = None


def create_app(runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="claude-agent-sdk 02-code-generation")

    app.state.settings = settings
    app.state.runner = runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        out = await generate(req.task, settings, app.state.runner)
        return RunResponse(
            solution=out.solution,
            tests=out.tests,
            summary=out.summary,
            tests_passed=out.tests_passed,
            files=out.files,
            tools_used=out.tool_calls,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            stop_reason=out.stop_reason,
            sandboxed=out.sandboxed,
            sandbox_note=out.sandbox_note,
        )

    return app


app = create_app()
