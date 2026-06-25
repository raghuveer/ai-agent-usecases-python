# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (langgraph). See langgraph/01-rag/README.md
"""Integration test for UC1 RAG (langgraph).

Hits the live local Qwen model via the gateway and a real Chroma index built
from the bundled corpus. Skipped unless RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run against the live local model",
)
def test_run_against_local_qwen(tmp_path):
    # Use a throwaway Chroma dir so the test is self-contained.
    os.environ.setdefault("CHROMA_DIR", str(tmp_path / "chroma"))
    app = create_app()
    with TestClient(app) as client:  # fires startup -> ingest + build graph
        resp = client.post(
            "/run",
            json={"question": "How many days do I have to return a product?"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert any(s["source"].endswith(".md") for s in body["sources"])
    # The relevant policy doc should surface for this question.
    assert any("returns" in s["source"] for s in body["sources"])
