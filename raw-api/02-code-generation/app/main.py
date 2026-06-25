# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (raw-api). See raw-api/02-code-generation/README.md
"""FastAPI app for UC2 code-generation, raw-api approach.

Wires an injectable LLM client on ``app.state`` so unit tests override it and run
fully offline. There is no index to build, so the lifespan is trivial.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import codegen, llm
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "02-code-generation"


class RunRequest(BaseModel):
    task: str = Field(max_length=8000)
    language: str | None = Field(default=None, max_length=40)


class RunResponse(BaseModel):
    code: str
    language: str
    explanation: str
    tests_passed: bool | None = None


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="raw-api 02-code-generation", lifespan=lifespan)

    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        language = req.language or settings.default_language
        result = codegen.generate(
            req.task,
            language=language,
            llm_call=_llm_call,
            run_code_check=settings.run_code_check,
            code_check_timeout=settings.code_check_timeout,
        )
        return RunResponse(**result)

    return app


app = create_app()
