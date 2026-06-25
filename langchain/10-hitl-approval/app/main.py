"""FastAPI app for UC10 hitl-approval (langchain approach).

``POST /run`` runs the draft chain to completion, then manually pauses: it stashes
the run in a ``RunRegistry`` and returns ``status=awaiting_approval``. ``POST
/resume`` looks the run up by ``run_id`` and continues (approved -> executed,
not approved -> rejected, unknown run_id -> 404).

The LLM/chain and the registry live on ``app.state``; unit tests inject a fake
LLM so nothing hits the network.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from . import hitl
from .llm import build_llm
from .settings import get_settings

APPROACH = "langchain"
USECASE = "10-hitl-approval"


class RunRequest(BaseModel):
    request: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    proposed_action: str


class ResumeRequest(BaseModel):
    run_id: str
    approved: bool
    feedback: str | None = None


class ResumeResponse(BaseModel):
    status: str
    result: str | None = None
    feedback: str | None = None


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "chain", None) is None:
            app.state.llm = llm or build_llm(settings)
            app.state.chain = hitl.build_draft_chain(app.state.llm, settings)
        yield

    app = FastAPI(title="langchain 10-hitl-approval", lifespan=lifespan)

    # Build eagerly when injected (tests use TestClient without lifespan context).
    if llm is not None:
        app.state.llm = llm
        app.state.chain = hitl.build_draft_chain(llm, settings)
    else:
        app.state.chain = None
    app.state.registry = hitl.RunRegistry()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        paused = hitl.start_run(
            req.request, chain=app.state.chain, registry=app.state.registry
        )
        return RunResponse(
            run_id=paused.run_id,
            status=paused.status,
            proposed_action=paused.proposed_action,
        )

    @app.post("/resume", response_model=ResumeResponse)
    def resume(req: ResumeRequest) -> ResumeResponse:
        try:
            out = hitl.resume_run(
                req.run_id,
                approved=req.approved,
                feedback=req.feedback,
                registry=app.state.registry,
            )
        except hitl.UnknownRunError:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return ResumeResponse(**out)

    return app


app = create_app()
