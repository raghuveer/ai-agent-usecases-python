# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""FastAPI app for UC8 autonomous-react, raw-api approach.

Wires an injectable LLM client onto ``app.state``; unit tests override it and
run fully offline. ``POST /run`` drives the hand-written ReAct loop.
"""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import llm, react, trace as trace_mod
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "08-autonomous-react"


class RunRequest(BaseModel):
    task: str = Field(max_length=8000)
    max_steps: int | None = Field(default=None, ge=1, le=12)


class StepModel(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class RunResponse(BaseModel):
    answer: str
    steps: list[StepModel]
    stopped_reason: str
    # Present only when the caller asks for it with `?trace=1`.
    # Schema: docs/trace-format.md
    trace: dict[str, Any] | None = None


def _sse(event: dict) -> str:
    """Format one loop event as a server-sent event frame.

    SSE is `event: <name>\\ndata: <json>\\n\\n`. The blank line terminates the
    frame — omit it and the client buffers forever waiting for more.
    """
    payload = dict(event)
    name = payload.pop("type")
    if "step" in payload:  # dataclass -> JSON
        payload["step"] = vars(payload["step"])
    if "result" in payload:
        result = payload.pop("result")
        payload = {
            "answer": result.answer,
            "stopped_reason": result.stopped_reason,
            "steps": [vars(s) for s in result.steps],
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _traced_tools(tools: react.ToolRegistry, tracer: trace_mod.Tracer):
    """Wrap a tool registry so every invocation records a span.

    Wrapping the registry rather than editing the loop keeps ``react.py`` free of
    tracing concerns — the loop still just calls whatever it was handed.
    """
    wrapped: react.ToolRegistry = {}
    for name, (fn, description) in tools.items():

        def traced(arg: str, _fn=fn, _name=name) -> str:
            started = time.perf_counter()
            output = _fn(arg)
            tracer.tool_span(
                name=_name,
                tool_input=arg,
                output=output,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return output

        wrapped[name] = (traced, description)
    return wrapped


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 08-autonomous-react")
    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _llm_call(messages: list[dict], tracer: trace_mod.Tracer | None = None) -> str:
        # Settings are read from app.state per request, not captured at startup,
        # so tests (and callers) can inject them the same way they inject the
        # client. They were already assigned there but previously ignored.
        settings = app.state.settings
        # Stop after the model's Action so it can't hallucinate the Observation.
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stop=["Observation:"],
            on_call=(
                None
                if tracer is None
                else lambda rec: tracer.llm_span(
                    messages=rec["messages"],
                    completion=rec["completion"],
                    duration_ms=rec["duration_ms"],
                    input_tokens=rec["input_tokens"],
                    output_tokens=rec["output_tokens"],
                    stop=rec["stop"],
                )
            ),
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Server-sent events: watch the loop reason, act and observe live.

        The raw-api version of streaming is exactly what it looks like — the
        OpenAI SDK yields deltas, the loop yields events, and this function
        formats SSE frames by hand. No framework is doing any of it, which is
        the point of this approach.

        Event names follow docs/streaming.md so all four approaches can be
        compared frame for frame.
        """
        settings = app.state.settings
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps

        def frames():
            def stream_call(messages: list[dict]):
                return llm.chat_stream(
                    app.state.client,
                    model=settings.llm_model,
                    messages=messages,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    stop=["Observation:"],
                )

            try:
                for event in react.iter_react(
                    req.task, llm_stream=stream_call, max_steps=max_steps
                ):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - the client must be told
                # A stream that dies silently looks identical to one that
                # finished, so failures are framed as events too.
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Run the loop. ``?trace=1`` returns what the agent actually did.

        Tracing is always available and never on by default: recording costs a
        few dict appends, and the trace echoes the caller's own prompt back to
        them rather than being written anywhere (see docs/trace-format.md).
        """
        settings = app.state.settings
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
        recording = trace or settings.trace_sink != "none"

        tracer = (
            trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                include_prompts=settings.trace_include_prompts,
            )
            if recording
            else None
        )

        result = react.run_react(
            req.task,
            llm_call=(lambda m: _llm_call(m, tracer)),
            tools=None if tracer is None else _traced_tools(react.TOOLS, tracer),
            max_steps=max_steps,
        )

        doc = None
        if tracer is not None:
            doc = tracer.finish(
                status="ok" if result.stopped_reason == "final_answer" else "capped",
                stop_reason=result.stopped_reason,
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            answer=result.answer,
            steps=[StepModel(**vars(s)) for s in result.steps],
            stopped_reason=result.stopped_reason,
            trace=doc if trace else None,
        )

    return app


app = create_app()
