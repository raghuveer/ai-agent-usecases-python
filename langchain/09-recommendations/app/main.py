"""FastAPI app for UC9 recommendations (langchain approach).

On startup it loads the bundled catalog + profiles and builds the LLM client.
Both are stored on ``app.state`` so unit tests inject fakes (the loaded data +
``FakeListChatModel``) and run with no network. The ranking is plain Python; only
the per-item reason runs an LCEL chain against the model.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .llm import build_llm
from .recommend import load_catalog, load_profiles, recommend
from .settings import get_settings

APPROACH = "langchain"
USECASE = "09-recommendations"


class RunRequest(BaseModel):
    user_id: str
    k: int | None = None


class Recommendation(BaseModel):
    item_id: str
    title: str
    reason: str


class RunResponse(BaseModel):
    recommendations: list[Recommendation] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "catalog", None) is None:
        app.state.catalog = load_catalog()
    if getattr(app.state, "profiles", None) is None:
        app.state.profiles = load_profiles()
    if getattr(app.state, "llm", None) is None:
        app.state.llm = build_llm(max_tokens=128)
    yield


app = FastAPI(title="langchain 09-recommendations", lifespan=lifespan)
# Pre-declare state so tests can set fakes before the first request.
app.state.catalog = None
app.state.profiles = None
app.state.llm = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "approach": APPROACH, "usecase": USECASE}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    settings = get_settings()
    llm: BaseChatModel = app.state.llm
    k = req.k if req.k is not None else settings.rec_top_k
    try:
        result = recommend(
            req.user_id,
            app.state.catalog,
            app.state.profiles,
            llm,
            settings.llm_model,
            k,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown user_id: {req.user_id}")
    return RunResponse(**result)
