# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langgraph). See langgraph/08-autonomous-react/README.md
"""FastAPI app for UC8 autonomous-react (langgraph approach).

``create_app`` accepts an injected LLM so unit tests run offline. ``POST /run``
invokes the ReAct StateGraph (reason → act → observe → loop | END).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from . import react as react_mod
from . import trace as trace_mod
from .llm import build_llm
from .react import run_react
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "08-autonomous-react"


def _sse(event: dict) -> str:
    """Format one graph event as a server-sent event frame. See docs/streaming.md."""
    payload = dict(event)
    name = payload.pop("type")
    if "result" in payload:
        payload = payload.pop("result")
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _traced_tools(tools: dict, tracer: trace_mod.Tracer) -> dict:
    """Wrap the tool registry so every invocation records a span.

    The tools are plain callables here (not LangChain ``Tool`` objects), so they
    emit no callbacks of their own — wrapping is how their spans get recorded,
    exactly as in `raw-api/08`. The graph itself stays untouched.
    """
    wrapped: dict = {}
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
    # Schema: docs/trace-format.md (plus `graph_path`, specific to this approach)
    trace: dict[str, Any] | None = None


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "llm", None) is None:
            app.state.llm = llm or build_llm(settings)
        yield

    app = FastAPI(title="langgraph 08-autonomous-react", lifespan=lifespan)
    # Inject eagerly when given (tests use TestClient without lifespan context).
    app.state.llm = llm

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run/stream")
    def run_stream(req: RunRequest) -> StreamingResponse:
        """Server-sent events — including `node` frames the others cannot emit.

        The other three approaches stream tokens. A graph can also stream
        **node transitions**, because the framework knows what a node is:
        ``stream_mode=["messages", "updates"]`` gives token chunks *and* each
        node's output as it completes. That is the live counterpart to
        `graph_path` in the trace. See docs/streaming.md.
        """
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps

        def frames():
            try:
                for event in react_mod.iter_react(
                    req.task, llm=app.state.llm, max_steps=max_steps
                ):
                    yield _sse(event)
            except Exception as exc:  # noqa: BLE001 - a silent stream looks finished
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest, trace: bool = False) -> RunResponse:
        """Run the graph. ``?trace=1`` returns what the agent actually did.

        The trace here also carries ``graph_path`` — the nodes actually visited.
        That is the thing this approach has and the others do not.
        """
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
        llm = app.state.llm
        tools = None

        recording = trace or settings.trace_sink != "none"
        tracer = None
        callbacks = None
        if recording:
            tracer = trace_mod.Tracer(
                approach=APPROACH,
                usecase=USECASE,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                include_prompts=settings.trace_include_prompts,
            )
            # One handler, passed through the graph's run config: it sees the
            # node transitions AND the model calls underneath them. Attaching it
            # to the model instead would record the calls but lose the route.
            callbacks = [trace_mod.TracingCallbackHandler(tracer)]
            tools = _traced_tools(react_mod.TOOLS, tracer)

        result = run_react(
            req.task,
            llm=llm,
            tools=tools,
            max_steps=max_steps,
            callbacks=callbacks,
        )

        doc = None
        if tracer is not None:
            doc = tracer.finish(
                status=(
                    "ok" if result["stopped_reason"] == "final_answer" else "capped"
                ),
                stop_reason=result["stopped_reason"],
            )
            if settings.trace_sink == "file":
                trace_mod.persist(doc, settings.trace_dir)

        return RunResponse(
            answer=result["answer"],
            steps=[StepModel(**s) for s in result["steps"]],
            stopped_reason=result["stopped_reason"],
            trace=doc if trace else None,
        )

    return app


app = create_app()
