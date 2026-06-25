# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langgraph). See langgraph/02-code-generation/README.md
"""FastAPI app for UC2 code-generation (langgraph approach).

On startup it builds the LLM client and compiles the codegen graph.
``create_app`` accepts an injected LLM so unit tests run offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .codegen import build_codegen_graph, run_codegen
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "02-code-generation"


class RunRequest(BaseModel):
    task: str
    language: str | None = None


class RunResponse(BaseModel):
    code: str
    language: str
    explanation: str
    tests_passed: bool | None = None


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "graph"):
            app.state.llm = llm or build_llm(settings)
            app.state.graph = build_codegen_graph(app.state.llm, settings)
        yield

    app = FastAPI(title="langgraph 02-code-generation", lifespan=lifespan)

    # When deps are injected (tests), build eagerly so requests work even if the
    # lifespan startup hasn't run (e.g. TestClient used without a context manager).
    if llm is not None:
        app.state.llm = llm
        app.state.graph = build_codegen_graph(llm, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        language = req.language or settings.default_language
        result = run_codegen(
            app.state.graph,
            req.task,
            language=language,
            run_code_check=settings.run_code_check,
        )
        return RunResponse(**result)

    return app


app = create_app()
