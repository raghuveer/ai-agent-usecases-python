"""FastAPI app for UC4 research-agent (langgraph approach).

On startup it loads the bundled corpus and builds the LLM client. ``create_app``
accepts an injected corpus + LLM so unit tests run offline.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from . import agent
from .llm import build_llm
from .settings import get_settings

APPROACH = "langgraph"
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


def create_app(
    corpus: agent.Corpus | None = None, llm: BaseChatModel | None = None
) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "corpus", None) is None:
            app.state.corpus = corpus or agent.Corpus()
        if getattr(app.state, "llm", None) is None:
            app.state.llm = llm or build_llm(settings)
        yield

    app = FastAPI(title="langgraph 04-research-agent", lifespan=lifespan)
    # Pre-set so injected deps work even without the lifespan (TestClient).
    app.state.corpus = corpus
    app.state.llm = llm

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
            llm=app.state.llm,
            max_steps=max_steps,
            top_k=settings.agent_top_k,
        )
        return RunResponse(
            answer=result.answer,
            sources=result.sources,
            steps=[StepModel(**s) for s in result.steps],
        )

    return app


app = create_app()
