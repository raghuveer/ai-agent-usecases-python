"""FastAPI app for UC1 RAG, raw-api approach.

Builds the Chroma index on startup and wires an injectable LLM client. Both the
retriever and the LLM call live on `app.state`, so unit tests override them and
run fully offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import llm, rag
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "01-rag"


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build the Chroma index once on boot (skip if a test pre-set one).
        if getattr(app.state, "retriever", None) is None:
            app.state.retriever = rag.ChromaRetriever(settings.chroma_dir)
        yield

    app = FastAPI(title="raw-api 01-rag", lifespan=lifespan)

    app.state.settings = settings
    app.state.client = llm.build_client(settings)
    # Default to None so the lifespan hook builds it; tests set it directly.
    app.state.retriever = None

    def _llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        top_k = req.top_k if req.top_k is not None else settings.rag_top_k
        result = rag.answer(
            req.question,
            retriever=app.state.retriever,
            llm_call=_llm_call,
            top_k=top_k,
        )
        return RunResponse(**result)

    return app


app = create_app()
