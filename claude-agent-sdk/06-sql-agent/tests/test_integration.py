# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""Integration test — a real agent discovers the schema and queries it.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1). Needs Node.js + the
Claude Code CLI.
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
def test_agent_discovers_schema_and_answers_from_rows():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={"question": "Which country has the most customers, and how many?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # It must have actually queried, not answered from thin air.
        assert body["queries"], "agent must run at least one SELECT"
        assert "run_select" in body["tools_used"]
        # Three US customers vs two UK in the seed data.
        assert "US" in body["answer"], body["answer"]
        assert "3" in body["answer"], body["answer"]


@pytest.mark.skipif(not RUN, reason=REASON)
def test_agent_recovers_when_it_writes_a_bad_query():
    """Errors come back as tool results, so the agent should self-correct."""
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "question": "What is the total revenue per product category? "
                "Join orders to products."
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["answer"].strip()
        assert body["queries"]
