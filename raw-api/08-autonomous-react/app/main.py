# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""FastAPI app for UC8 autonomous-react, raw-api approach.

Wires an injectable LLM client onto ``app.state``; unit tests override it and
run fully offline. ``POST /run`` drives the hand-written ReAct loop.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import llm, react
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


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 08-autonomous-react")
    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _llm_call(messages: list[dict]) -> str:
        # Stop after the model's Action so it can't hallucinate the Observation.
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stop=["Observation:"],
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
        result = react.run_react(
            req.task,
            llm_call=_llm_call,
            max_steps=max_steps,
        )
        return RunResponse(
            answer=result.answer,
            steps=[StepModel(**vars(s)) for s in result.steps],
            stopped_reason=result.stopped_reason,
        )

    return app


app = create_app()
