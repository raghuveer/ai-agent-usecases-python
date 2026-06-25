# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langchain). See langchain/05-support-triage/README.md
"""FastAPI app for UC5 support-triage (langchain approach).

On startup it builds the LLM client and stores it on ``app.state`` so unit tests
inject a ``FakeListChatModel`` and run with no network. The triage flow
(classify -> route -> respond -> escalate) lives in ``triage.py``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from . import triage
from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "05-support-triage"


class RunRequest(BaseModel):
    message: str
    session_id: str | None = None


class RunResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    escalate: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=384)
    yield


app = FastAPI(title="langchain 05-support-triage", lifespan=lifespan)
# Pre-declare state so tests can set a fake before the first request.
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    result = triage.triage(
        req.message,
        session_id=req.session_id,
        llm=llm,
        escalate_threshold=settings.escalate_threshold,
    )
    return RunResponse(**result)
