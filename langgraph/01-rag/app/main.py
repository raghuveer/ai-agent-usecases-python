"""FastAPI app for UC1 RAG (langgraph approach).

On startup it ingests the bundled corpus into Chroma and compiles the RAG graph.
``create_app`` accepts an injected retriever + LLM so unit tests run offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .llm import build_llm
from .rag import Retriever, build_rag_graph, ingest_corpus, sources_from_docs
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "01-rag"


class RunRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    source: str
    snippet: str


class RunResponse(BaseModel):
    answer: str
    sources: list[Source] = []


def create_app(retriever: Retriever | None = None,
               llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build any deps not already injected (production path ingests Chroma).
        if not hasattr(app.state, "graph"):
            app.state.retriever = retriever or ingest_corpus(settings)
            app.state.llm = llm or build_llm(settings)
            app.state.graph = build_rag_graph(app.state.retriever, app.state.llm, settings)
        yield

    app = FastAPI(title="langgraph 01-rag", lifespan=lifespan)

    # When deps are injected (tests), build eagerly so requests work even if the
    # lifespan startup hasn't run (e.g. TestClient used without a context manager).
    if retriever is not None and llm is not None:
        app.state.retriever = retriever
        app.state.llm = llm
        app.state.graph = build_rag_graph(retriever, llm, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        top_k = req.top_k if req.top_k is not None else settings.rag_top_k
        out = app.state.graph.invoke({
            "question": req.question,
            "top_k": top_k,
            "documents": [],
            "answer": "",
        })
        return RunResponse(
            answer=out["answer"],
            sources=[Source(**s) for s in sources_from_docs(out["documents"])],
        )

    return app


app = create_app()
