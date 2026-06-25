# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langgraph). See langgraph/06-sql-agent/README.md
"""FastAPI app for UC6 sql-agent (langgraph approach).

On startup it builds a tiny SQLite DB from ``data/seed.sql`` and compiles the SQL
agent graph. ``create_app`` accepts an injected DB connection + LLM so unit tests
run offline.

Pipeline (as graph edges): generate -> validate -> (execute | reject). A
validation rejection surfaces as HTTP 400; anything that is not a single
read-only SELECT is rejected before the DB is touched.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .llm import build_llm
from .settings import get_settings
from .sqlagent import build_db, build_sql_graph, initial_state

APPROACH = "langgraph"
USECASE = "06-sql-agent"


class RunRequest(BaseModel):
    question: str = Field(max_length=8000)


class RunResponse(BaseModel):
    sql: str
    rows: list = []
    explanation: str


def create_app(conn=None, llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "graph"):
            app.state.conn = conn or build_db()
            app.state.llm = llm or build_llm(settings)
            app.state.graph = build_sql_graph(app.state.conn, app.state.llm, settings)
        yield
        c = getattr(app.state, "conn", None)
        if c is not None:
            c.close()

    app = FastAPI(title="langgraph 06-sql-agent", lifespan=lifespan)

    # When deps are injected (tests), build eagerly so requests work even if the
    # lifespan startup hasn't run (e.g. TestClient used without a context manager).
    if conn is not None and llm is not None:
        app.state.conn = conn
        app.state.llm = llm
        app.state.graph = build_sql_graph(conn, llm, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        out = app.state.graph.invoke(initial_state(req.question))
        if out.get("error"):
            raise HTTPException(status_code=400, detail=out["error"])
        return RunResponse(
            sql=out["sql"], rows=out["rows"], explanation=out["explanation"]
        )

    return app


app = create_app()
