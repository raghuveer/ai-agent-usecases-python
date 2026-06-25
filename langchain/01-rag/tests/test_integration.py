# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (langchain). See langchain/01-rag/README.md
"""Integration test — hits the live local Qwen via the gateway + real Chroma.

Gated: skipped unless ``RUN_INTEGRATION=1``. Uses chromadb default embeddings
(no torch). Keep ``max_tokens`` small.
"""
from __future__ import annotations

import os

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]


def test_rag_roundtrip_against_local_qwen(tmp_path):
    from app.llm import build_llm
    from app.rag import answer_question, build_vectorstore
    from app.settings import get_settings

    settings = get_settings()
    # Use an isolated, throwaway chroma dir for the test run.
    settings.chroma_dir = str(tmp_path / "chroma")

    store = build_vectorstore(settings)
    retriever = store.as_retriever(search_kwargs={"k": settings.rag_top_k})
    llm = build_llm(settings, max_tokens=256)

    answer, sources = answer_question(
        "How many days do I have to return a product?",
        retriever,
        llm,
        settings.llm_model,
    )

    assert isinstance(answer, str) and answer.strip()
    assert sources, "expected at least one retrieved source"
    # The corpus says 30 days; the grounded model should mention it.
    assert "30" in answer
