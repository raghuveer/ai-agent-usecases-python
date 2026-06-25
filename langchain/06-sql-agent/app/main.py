"""FastAPI app for UC6 sql-agent (langchain approach).

On startup it builds a tiny SQLite DB from ``data/seed.sql`` and the LLM client.
Both are stored on ``app.state`` so unit tests inject a fake DB +
``FakeListChatModel`` and run with no network.

Pipeline: schema-injection LCEL chain -> ONE generated SQL -> strict read-only
SELECT validation (writes/DDL/multi-statement -> HTTP 400) -> read-only execute
-> NL summary.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .llm import build_llm
from .settings import get_settings
from .sqlagent import SQLValidationError, answer_question, build_db

APPROACH = "langchain"
USECASE = "06-sql-agent"


class RunRequest(BaseModel):
    question: str


class RunResponse(BaseModel):
    sql: str
    rows: list = []
    explanation: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "conn", None) is None:
        app.state.conn = build_db()
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=256)
    yield
    conn = getattr(app.state, "conn", None)
    if conn is not None:
        conn.close()


app = FastAPI(title="langchain 06-sql-agent", lifespan=lifespan)
# Pre-declare state so tests can set fakes before the first request.
app.state.conn = None
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    try:
        result = answer_question(req.question, app.state.conn, llm, settings.llm_model)
    except SQLValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RunResponse(**result)
