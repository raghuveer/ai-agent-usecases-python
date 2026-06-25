# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langgraph). See langgraph/03-data-extraction/README.md
"""FastAPI app for UC3 data-extraction (langgraph approach).

On startup it builds the LLM client and compiles the extraction graph.
``create_app`` accepts an injected LLM so unit tests run offline. ``POST /run``
extracts a fixed ``Invoice`` schema from raw text via the graph (extract ->
validate -> retry-once), and returns 422 if still invalid.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .extract import ExtractionError, Invoice, extract_invoice
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "03-data-extraction"


class RunRequest(BaseModel):
    text: str = Field(max_length=20000)


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "llm", None) is None:
            app.state.llm = llm or build_llm(settings)
        yield

    app = FastAPI(title="langgraph 03-data-extraction", lifespan=lifespan)

    # When an LLM is injected (tests), set it eagerly so requests work even if
    # the lifespan startup hasn't run (TestClient without a context manager).
    app.state.llm = llm

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run")
    def run(req: RunRequest) -> dict:
        try:
            invoice: Invoice = extract_invoice(
                req.text, app.state.llm, settings
            )
        except ExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": str(exc), "valid": False, "raw": exc.raw},
            )
        return {**invoice.model_dump(), "valid": True}

    return app


app = create_app()
