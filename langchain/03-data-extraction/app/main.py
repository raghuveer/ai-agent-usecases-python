"""FastAPI app for UC3 data-extraction (langchain approach).

On startup it builds the LLM client and stores it on ``app.state`` so unit tests
inject a ``FakeListChatModel`` and run with no network. ``POST /run`` extracts a
fixed ``Invoice`` schema from raw text via an LCEL chain, validates with
Pydantic, and retries the chain once on failure.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .extract import ExtractionError, Invoice, extract_invoice
from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "03-data-extraction"


class RunRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=512)
    yield


app = FastAPI(title="langchain 03-data-extraction", lifespan=lifespan)
# Pre-declare state so tests can set a fake LLM before the first request.
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run")
def run(req: RunRequest) -> dict:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    try:
        invoice: Invoice = extract_invoice(req.text, llm, settings.llm_model)
    except ExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": str(exc), "valid": False, "raw": exc.raw},
        )
    return {**invoice.model_dump(), "valid": True}
