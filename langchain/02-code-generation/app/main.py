# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langchain). See langchain/02-code-generation/README.md
"""FastAPI app for UC2 code-generation (langchain approach).

On startup it builds the LLM client and stores it on ``app.state`` so unit tests
inject a ``FakeListChatModel`` and run with no network. There is no index to
build, so the lifespan only wires the LLM.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from . import codegen
from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "02-code-generation"


class RunRequest(BaseModel):
    task: str = Field(max_length=8000)
    language: str | None = Field(default=None, max_length=40)


class RunResponse(BaseModel):
    code: str
    language: str
    explanation: str
    tests_passed: bool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=1024)
    yield


app = FastAPI(title="langchain 02-code-generation", lifespan=lifespan)
# Pre-declare state so tests can set a fake before the first request.
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    language = req.language or settings.default_language

    result = codegen.generate(
        req.task,
        language=language,
        llm=llm,
        model_name=settings.llm_model,
        run_code_check=settings.run_code_check,
        code_check_timeout=settings.code_check_timeout,
    )
    return RunResponse(**result)
