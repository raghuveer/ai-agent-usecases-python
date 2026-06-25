# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (raw-api). See raw-api/09-recommendations/README.md
"""FastAPI app for UC9 recommendations, raw-api approach.

Loads the bundled catalog + profiles on startup and wires an injectable LLM
client. The deterministic ranking is plain Python; only the per-item reason
calls the model. Catalog/profiles and the LLM call live on ``app.state``, so
unit tests override them and run fully offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import llm, recommend
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "09-recommendations"


class RunRequest(BaseModel):
    user_id: str = Field(max_length=200)
    k: int | None = Field(default=None, ge=1, le=50)


class Recommendation(BaseModel):
    item_id: str
    title: str
    reason: str


class RunResponse(BaseModel):
    recommendations: list[Recommendation]


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the bundled data once on boot (skip if a test pre-set it).
        if getattr(app.state, "catalog", None) is None:
            app.state.catalog = recommend.load_catalog()
        if getattr(app.state, "profiles", None) is None:
            app.state.profiles = recommend.load_profiles()
        yield

    app = FastAPI(title="raw-api 09-recommendations", lifespan=lifespan)

    app.state.settings = settings
    app.state.client = llm.build_client(settings)
    # Default to None so the lifespan hook loads them; tests set them directly.
    app.state.catalog = None
    app.state.profiles = None

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
        k = req.k if req.k is not None else settings.rec_top_k
        try:
            result = recommend.recommend(
                req.user_id,
                catalog=app.state.catalog,
                profiles=app.state.profiles,
                llm_call=_llm_call,
                k=k,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown user_id: {req.user_id}")
        return RunResponse(**result)

    return app


app = create_app()
