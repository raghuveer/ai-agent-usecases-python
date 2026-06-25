"""FastAPI app for UC1 RAG (langchain approach).

On startup it builds/loads the Chroma index and a retriever, and builds the LLM
client. Both are stored on ``app.state`` so unit tests inject fakes (a fake
retriever + ``FakeListChatModel``) and run with no network.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from .llm import build_llm
from .rag import answer_question, build_vectorstore
from .settings import get_settings

APPROACH = "langchain"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if getattr(app.state, "retriever", None) is None:
        store = build_vectorstore(settings)
        app.state.retriever = store.as_retriever(
            search_kwargs={"k": settings.rag_top_k}
        )
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=512)
    yield


app = FastAPI(title="langchain 01-rag", lifespan=lifespan)
# Pre-declare state so tests can set fakes before the first request.
app.state.retriever = None
app.state.llm = None


def _snippet(doc: Document, limit: int = 240) -> str:
    text = doc.page_content.strip().replace("\n", " ")
    return text[:limit]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    retriever: BaseRetriever = app.state.retriever
    llm: BaseChatModel = app.state.llm

    # Honour a per-request top_k override when the retriever supports it.
    if req.top_k is not None:
        try:
            retriever = retriever.__class__(
                **{**retriever.__dict__, "search_kwargs": {"k": req.top_k}}
            )
        except Exception:
            pass  # fake/custom retrievers: fall back to the default one.

    answer, docs = answer_question(
        req.question, retriever, llm, settings.llm_model
    )
    sources = [
        Source(source=d.metadata.get("source", "?"), snippet=_snippet(d))
        for d in docs
    ]
    return RunResponse(answer=answer, sources=sources)
