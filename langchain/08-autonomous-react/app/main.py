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

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .llm import build_llm
from .react import run_react
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm()
    yield


app = FastAPI(title="langchain 08-autonomous-react", lifespan=lifespan)
# Pre-declare state so tests can inject a fake before the first request.
app.state.llm = None
app.state.tools = build_tools()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    max_steps = req.max_steps if req.max_steps is not None else settings.max_steps
    llm: BaseChatModel = app.state.llm
    result = run_react(
        req.task,
        llm=llm,
        tools=app.state.tools,
        max_steps=max_steps,
    )
    return RunResponse(
        answer=result.answer,
        steps=[StepModel(**vars(s)) for s in result.steps],
        stopped_reason=result.stopped_reason,
    )
