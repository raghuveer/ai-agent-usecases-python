# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langchain). See langchain/08-autonomous-react/README.md
"""FastAPI app for UC8 autonomous-react (langchain approach).

Builds the LangChain ``Tool`` set and an injectable chat model on ``app.state``;
unit tests set a ``FakeListChatModel`` and run offline. ``POST /run`` drives the
text-ReAct agent loop.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

from . import trace as trace_mod
from .llm import build_llm
from .react import iter_react, run_react
from .settings import get_settings
from .tools import build_tools

APPROACH = "langchain"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm()
    yield


app = FastAPI(title="langchain 08-autonomous-react", lifespan=lifespan)
# Pre-declare state so tests can inject a fake before the first request.
app.state.llm = None
app.state.tools = build_tools()


def _sse(event: dict) -> str:
    """Format one loop event as a server-sent event frame. See docs/streaming.md."""
    payload = dict(event)
    name = payload.pop("type")
    if "step" in payload:
        payload["step"] = vars(payload["step"])
    if "result" in payload:
        result = payload.pop("result")
        payload = {
            "answer": result.answer,
            "stopped_reason": result.stopped_reason,
            "steps": [vars(s) for s in result.steps],
        }
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run/stream")
def run_stream(req: RunRequest) -> StreamingResponse:
    """Server-sent events: watch the loop reason, act and observe live.

    The langchain difference is one line inside the loop: ``llm.stream()``
    instead of ``llm.invoke()``. The framework hands you an iterator of
    ``AIMessageChunk``s — no SSE parsing, no delta bookkeeping. Compare with
    `raw-api/08`, which does that itself.
    """
    settings = get_settings()
    max_steps = req.max_steps if req.max_steps is not None else settings.max_steps

    def frames():
        try:
            for event in iter_react(
                req.task,
                llm=app.state.llm,
                tools=app.state.tools,
                max_steps=max_steps,
                stream=True,
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
    """Run the loop. ``?trace=1`` returns what the agent actually did.

    Note how little wiring this needs compared with `raw-api/08`: LangChain
    already has an observation seam, so the tracer attaches as a callback and
    the ReAct loop is untouched. See docs/trace-format.md.
    """
    settings = get_settings()
    max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
    llm: BaseChatModel = app.state.llm
    tools = app.state.tools

    recording = trace or settings.trace_sink != "none"
    tracer = handler = None
    if recording:
        tracer = trace_mod.Tracer(
            approach=APPROACH,
            usecase=USECASE,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            include_prompts=settings.trace_include_prompts,
        )
        handler = trace_mod.TracingCallbackHandler(tracer)
        llm = llm.with_config(callbacks=[handler])
        # Rebuilt rather than reconfigured: `with_config` on a Tool returns a
        # RunnableBinding, which no longer carries the `.name` the loop
        # dispatches on.
        tools = [
            Tool(
                name=t.name,
                func=t.func,
                description=t.description,
                callbacks=[handler],
            )
            for t in tools
        ]

    result = run_react(
        req.task,
        llm=llm,
        tools=tools,
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
