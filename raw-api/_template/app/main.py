"""Minimal FastAPI skeleton for the raw-api approach (Phase 0 template).

`GET /health` reports the approach/usecase. `POST /run` echoes the question
back through the (mockable) LLM and returns `{"answer": ..., "sources": []}`.
Copy this folder as the starting point for a new use case.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from . import llm
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "_template"


class RunRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    source: str
    snippet: str


class RunResponse(BaseModel):
    answer: str
    sources: list[Source]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="raw-api template")

    # Injectable client: built once, stored on app state so tests can override.
    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        answer = llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt="You are a terse echo assistant. Repeat the user's message verbatim.",
            user_prompt=req.question,
            max_tokens=128,
        )
        return RunResponse(answer=answer, sources=[])

    return app


app = create_app()
