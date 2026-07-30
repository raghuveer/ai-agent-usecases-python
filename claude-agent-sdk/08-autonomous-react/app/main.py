# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC08 Autonomous ReAct (claude-agent-sdk). See claude-agent-sdk/08-autonomous-react/README.md
"""FastAPI app for UC08 autonomous-react (claude-agent-sdk approach) — the showcase."""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import trace as trace_mod
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
    # `steps`, not `trace`: this shipped as `trace` through v0.4.0, which
    # collided with the shared trace document and made the four approaches
    # disagree on field names for the same idea. Renamed in v0.5.0 — comparing
    # approaches is the point of the repo, so their responses should line up.
    steps: list[StepModel]
    num_turns: int
    cost_usd: float
    hit_turn_limit: bool
    # The shared trace document (docs/trace-format.md), present only with
    # `?trace=1` — same field name as the other three approaches.
    trace: dict[str, Any] | None = None


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
    async def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Run the agent. ``?trace=1`` returns what the SDK was willing to tell us.

        Deliberately less than the other three approaches can report — see
        app/trace.py and the `not_captured` list in the response.
        """
        settings = app.state.settings
        started = time.perf_counter()
        out = await run_react(req.question, settings, app.state.runner)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        doc = None
        if trace or settings.trace_sink != "none":
            tracer = trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                max_turns=settings.agent_max_turns,
                max_budget_usd=settings.agent_max_budget_usd,
                include_prompts=settings.trace_include_prompts,
            )
            for step in out.trace:
                tracer.tool_span(name=step.tool, tool_input=step.input)
            doc = tracer.finish(
                status="capped" if out.hit_turn_limit else "ok",
                stop_reason="max_turns" if out.hit_turn_limit else "final_answer",
                num_turns=out.num_turns,
                cost_usd=out.cost_usd,
                duration_ms=elapsed_ms,
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            answer=out.answer,
            steps=[StepModel(tool=s.tool, input=s.input) for s in out.trace],
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            hit_turn_limit=out.hit_turn_limit,
            trace=doc if trace else None,
        )

    return app


app = create_app()
