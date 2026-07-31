# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""FastAPI app for UC07 multi-agent (claude-agent-sdk approach) — the showcase."""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import Runner, build_options, iter_events, outcome_of
from . import trace as trace_mod
from .settings import get_settings
from .team import CORPUS_DIR, LEAD_PROMPT, TEAM, TEAM_TOOLS, run_team

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
    stop_reason: str
    # Present only with `?trace=1`. Schema: docs/trace-format.md
    trace: dict[str, Any] | None = None


def _sse(event: dict) -> str:
    """Format one team event as a server-sent event frame."""
    payload = dict(event)
    name = payload.pop("type")
    if "result" in payload:
        result = payload.pop("result")
        payload = {
            "report": result.text,
            "num_turns": result.num_turns,
            "cost_usd": result.cost_usd,
            # Mapped, not raw: the other three approaches end their stream with
            # `final_answer` / `max_steps`, and a client watching all four should
            # not have to special-case Anthropic's `end_turn` here.
            "stopped_reason": trace_mod.to_trace_reason(outcome_of(result)),
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


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

    @app.post("/run/stream")
    async def run_stream(req: RunRequest) -> StreamingResponse:
        """Server-sent events — the `step` frames are the delegations.

        Each one is an `Agent` tool call carrying a `subagent_type`, so you watch
        the lead hand work to researcher / analyst / writer as it happens. What
        you still cannot see is *inside* a subagent: the harness runs it and
        hands back only the result. Nor are there `token` frames — the SDK
        yields whole turns. See docs/streaming.md.
        """
        settings = app.state.settings

        async def frames():
            try:
                options = build_options(
                    settings,
                    system_prompt=LEAD_PROMPT,
                    allowed_tools=TEAM_TOOLS,
                    tools=TEAM_TOOLS,
                    agents=TEAM,
                    cwd=str(CORPUS_DIR),
                    max_turns=max(settings.agent_max_turns, 20),
                )
                async for event in iter_events(req.question, options):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - a silent stream looks finished
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Run the team. ``?trace=1`` records the delegations the lead made.

        As in UC08 this trace is deliberately partial — the SDK reports tool
        calls and run totals, not messages or tokens. See app/trace.py.
        """
        settings = app.state.settings
        started = time.perf_counter()
        out = await run_team(req.question, settings, app.state.runner)
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
            for name in out.tools_used:
                tracer.tool_span(name=name, tool_input={})
            # Was inferred from whether the report came back empty, which named
            # "max_turns" for every early stop — including a budget cap or an
            # error. The run now reports its own reason.
            reason = trace_mod.to_trace_reason(out.stop_reason)
            doc = tracer.finish(
                status=trace_mod.status_for(reason),
                stop_reason=reason,
                num_turns=out.num_turns,
                cost_usd=out.cost_usd,
                duration_ms=elapsed_ms,
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            report=out.report,
            subagents_used=out.subagents_used,
            tools_used=out.tools_used,
            num_turns=out.num_turns,
            cost_usd=out.cost_usd,
            stop_reason=out.stop_reason,
            trace=doc if trace else None,
        )

    return app


app = create_app()
