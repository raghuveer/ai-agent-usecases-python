# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langchain project template. See langchain/_template/README.md
"""Minimal FastAPI skeleton for the langchain approach (Phase 0 template).

This is the starting point other use cases copy. It wires up:
- ``GET /health`` — liveness + identity.
- ``POST /run`` — sends the question to the (mockable) LLM and echoes the reply.

The LLM client is built lazily and stored on ``app.state`` so tests can inject a
fake model. There is no retrieval here; that is added per use case.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "_template"


class RunRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    source: str
    snippet: str


class RunResponse(BaseModel):
    answer: str
    sources: list[Source] = []


def _system_prompt(model: str) -> str:
    """Base system prompt, disabling qwen3 thinking when relevant."""
    base = "You are a helpful assistant. Answer concisely."
    if model.startswith("qwen3"):
        return "/no_think\n" + base
    return base


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "llm") or app.state.llm is None:
        app.state.llm = build_llm(max_tokens=256)
    yield


app = FastAPI(title="langchain template", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    llm: BaseChatModel = app.state.llm
    settings = get_settings()
    messages = [
        SystemMessage(content=_system_prompt(settings.llm_model)),
        HumanMessage(content=req.question),
    ]
    result = llm.invoke(messages)
    return RunResponse(answer=result.content, sources=[])
