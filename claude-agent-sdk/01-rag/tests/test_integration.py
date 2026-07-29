# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC01 Q&A / RAG (claude-agent-sdk). See claude-agent-sdk/01-rag/README.md
"""Integration test — a real agent searches the corpus and answers from it.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1), capped by
AGENT_MAX_TURNS / AGENT_MAX_BUDGET_USD. Needs Node.js + the Claude Code CLI.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1" and os.getenv("RUN_ANTHROPIC_TESTS") == "1"
REASON = "set RUN_INTEGRATION=1 and RUN_ANTHROPIC_TESTS=1 to run live agent tests"


@pytest.mark.skipif(not RUN, reason=REASON)
def test_agent_retrieves_from_corpus_and_cites_it():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={"question": "Could small local models drive a text ReAct loop?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["answer"].strip()
        # Grounded means it actually opened a document.
        assert body["sources"], "answer must be backed by a file the agent read"
        assert "model-findings.md" in body["sources"], body["sources"]


@pytest.mark.skipif(not RUN, reason=REASON)
def test_agent_admits_when_the_corpus_lacks_the_answer():
    """Lexical retrieval finds nothing here; it must say so, not confabulate."""
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={"question": "What is the melting point of tungsten?"},
        )
        assert resp.status_code == 200, resp.text
        answer = resp.json()["answer"].lower()
        assert any(
            phrase in answer
            for phrase in ("does not", "doesn't", "no information", "not contain", "cannot")
        ), resp.json()["answer"]
