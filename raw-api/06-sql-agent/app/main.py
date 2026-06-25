# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (raw-api). See raw-api/06-sql-agent/README.md
"""FastAPI app for UC6 sql-agent, raw-api approach.

Builds a tiny SQLite DB from ``data/seed.sql`` on startup and wires an injectable
LLM client. The DB connection and the LLM call live on ``app.state`` so unit
tests override them and run fully offline.

Pipeline: schema-injection prompt -> ONE generated SQL -> strict read-only SELECT
validation (writes/DDL/multi-statement -> HTTP 400) -> read-only execute -> NL
summary.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import llm, sqlagent
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "06-sql-agent"


class RunRequest(BaseModel):
    question: str


class RunResponse(BaseModel):
    sql: str
    rows: list
    explanation: str


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build the SQLite DB once on boot (skip if a test pre-set one).
        if getattr(app.state, "conn", None) is None:
            app.state.conn = sqlagent.build_db()
        yield
        conn = getattr(app.state, "conn", None)
        if conn is not None:
            conn.close()

    app = FastAPI(title="raw-api 06-sql-agent", lifespan=lifespan)

    app.state.settings = settings
    app.state.client = llm.build_client(settings)
    # Default to None so the lifespan hook builds it; tests set it directly.
    app.state.conn = None

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
        try:
            result = sqlagent.answer(
                req.question,
                conn=app.state.conn,
                llm_call=_llm_call,
            )
        except sqlagent.SQLValidationError as exc:
            # Generated statement was not a single read-only SELECT.
            raise HTTPException(status_code=400, detail=str(exc))
        return RunResponse(**result)

    return app


app = create_app()
