"""FastAPI app for UC5 support-triage, raw-api approach.

Wires an injectable LLM client onto ``app.state``. Unit tests override the
client with a stub and run fully offline. The triage flow (classify -> route ->
respond -> escalate) lives in ``triage.py``.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from . import llm, triage
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "05-support-triage"


class RunRequest(BaseModel):
    message: str
    session_id: str | None = None


class RunResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    escalate: bool


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 05-support-triage")

    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    def _classify_call(system_prompt: str, user_prompt: str) -> str:
        # Deterministic, short JSON classification.
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=64,
        )

    def _respond_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            app.state.client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=384,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        result = triage.triage(
            req.message,
            session_id=req.session_id,
            classify_call=_classify_call,
            respond_call=_respond_call,
            escalate_threshold=settings.escalate_threshold,
        )
        return RunResponse(**result)

    return app


app = create_app()
