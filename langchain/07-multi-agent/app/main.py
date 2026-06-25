# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langchain). See langchain/07-multi-agent/README.md
"""FastAPI app for UC7 multi-agent (langchain approach).

Builds an injectable chat model on ``app.state``; unit tests set a
``FakeListChatModel`` and run offline. ``POST /run`` drives the role-chain
orchestrator (researcher → writer → reviewer, with one revise loop).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .agents import orchestrate
from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "07-multi-agent"


class RunRequest(BaseModel):
    topic: str = Field(max_length=8000)


class Contributions(BaseModel):
    research: str
    writer: str
    reviewer: str


class RunResponse(BaseModel):
    draft: str
    review: str
    approved: bool
    contributions: Contributions


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm()
    yield


app = FastAPI(title="langchain 07-multi-agent", lifespan=lifespan)
# Pre-declare state so tests can inject a fake before the first request.
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    result = orchestrate(
        req.topic,
        llm=llm,
        research_top_k=settings.research_top_k,
        max_revisions=settings.max_revisions,
    )
    return RunResponse(
        draft=result.draft,
        review=result.review,
        approved=result.approved,
        contributions=Contributions(**result.contributions),
    )
