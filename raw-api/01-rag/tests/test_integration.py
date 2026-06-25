# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (raw-api). See raw-api/01-rag/README.md
"""Integration test — hits the live local model via the gateway.

Builds the real Chroma index from the bundled corpus and calls the LLM. Gated:
runs only when RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from app import llm, rag
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_rag_round_trip_against_local_qwen():
    settings = get_settings()
    client = llm.build_client(settings)

    tmp = tempfile.mkdtemp()
    try:
        retriever = rag.ChromaRetriever(tmp)

        def llm_call(system_prompt: str, user_prompt: str) -> str:
            return llm.chat(
                client,
                model=settings.llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        result = rag.answer(
            "How many days do I have to return a product?",
            retriever=retriever,
            llm_call=llm_call,
            top_k=settings.rag_top_k,
        )
        retriever.close()
    finally:
        # Windows keeps Chroma's segment files briefly locked; tolerate that.
        shutil.rmtree(tmp, ignore_errors=True)

    assert result["answer"].strip()
    assert result["sources"]
    # The returns doc should be among the retrieved sources for this question.
    assert any("returns" in s["source"] for s in result["sources"])
