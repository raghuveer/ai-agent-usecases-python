# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""Integration test — a real research agent works the bundled corpus.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1). Deliberately exercises
**offline** mode so the assertions are deterministic and the test runs on an
air-gapped host; the web path is opt-in via RESEARCH_ALLOW_WEB=1 and is not
asserted here because live search results are not reproducible.
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
def test_offline_research_answers_from_corpus_with_citations():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "question": "What did we learn about small local models and "
                "multi-step tool use, and how did the gateway affect it?"
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["mode"] == "offline"
        assert body["answer"].strip()
        assert body["citations"], "a researched answer must cite what it opened"
        # It should have consulted more than one note for a two-part question.
        assert len(body["citations"]) >= 2, body["citations"]
        assert not any(c.startswith("http") for c in body["citations"]), (
            "offline mode must not produce web citations"
        )
