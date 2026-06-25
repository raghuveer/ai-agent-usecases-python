# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (raw-api). See raw-api/03-data-extraction/README.md
"""FastAPI app for UC3 data-extraction, raw-api approach.

Wires an injectable LLM client onto ``app.state`` so unit tests override it and
run fully offline. ``POST /run`` extracts a fixed ``Invoice`` schema from raw
text, validates with Pydantic, and retries the model once on failure.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import extract, llm
from .extract import ExtractionError, Invoice
from .settings import get_settings

APPROACH = "raw-api"
USECASE = "03-data-extraction"


class RunRequest(BaseModel):
    text: str = Field(max_length=20000)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="raw-api 03-data-extraction")

    app.state.settings = settings
    app.state.client = llm.build_client(settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run")
    def run(req: RunRequest) -> dict:
        # Read settings/client off app.state so tests can override either.
        cfg = app.state.settings
        # "native" mode asks the provider for guaranteed JSON (OpenAI-compatible
        # JSON mode); "text" (default) prompts for JSON and parses it from the
        # reply — byte-identical to before, no response_format sent.
        native = cfg.llm_structured_mode == "native"
        response_format = {"type": "json_object"} if native else None

        def _llm_call(system_prompt: str, user_prompt: str) -> str:
            return llm.chat(
                app.state.client,
                model=cfg.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                response_format=response_format,
            )

        try:
            invoice: Invoice = extract.extract_invoice(
                req.text,
                llm_call=_llm_call,
                mode=cfg.llm_structured_mode,
            )
        except ExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": str(exc), "valid": False, "raw": exc.raw},
            )
        return {**invoice.model_dump(), "valid": True}

    return app


app = create_app()
