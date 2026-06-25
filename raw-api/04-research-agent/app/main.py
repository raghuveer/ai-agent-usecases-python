# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (raw-api). See raw-api/04-research-agent/README.md
"""FastAPI app for UC4 research-agent, raw-api approach.

Loads the bundled corpus on startup and wires an injectable LLM client. The
corpus and the LLM call both live on ``app.state`` so unit tests override them
and run fully offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import agent, llm
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "04-research-agent"


class RunRequest(BaseModel):
    question: str
    max_steps: int | None = None


class StepModel(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class RunResponse(BaseModel):
    answer: str
    sources: list[str] = []
    steps: list[StepModel] = []


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "corpus", None) is None:
            app.state.corpus = agent.Corpus()
        yield

    app = FastAPI(title="raw-api 04-research-agent", lifespan=lifespan)

    app.state.settings = settings
    app.state.client = llm.build_client(settings)
    # Default to None so the lifespan hook builds it; tests set it directly.
    app.state.corpus = None

    def _llm_call(user_prompt: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=agent.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        max_steps = (
            req.max_steps if req.max_steps is not None else settings.agent_max_steps
        )
        result = agent.run_agent(
            req.question,
            corpus=app.state.corpus,
            llm_call=_llm_call,
            max_steps=max_steps,
            top_k=settings.agent_top_k,
        )
        return RunResponse(
            answer=result.answer,
            sources=result.sources,
            steps=[
                StepModel(
                    thought=s.thought,
                    action=s.action,
                    action_input=s.action_input,
                    observation=s.observation,
                )
                for s in result.steps
            ],
        )

    return app


app = create_app()
