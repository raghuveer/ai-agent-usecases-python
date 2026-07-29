# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""Integration test — a real lead agent delegates to real subagents.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1). This is the most
expensive example in the approach: every delegation spawns a subagent with its
own context. AGENT_MAX_BUDGET_USD is the backstop.
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
def test_lead_delegates_and_returns_a_report():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "question": "What did we learn about small local models driving "
                "ReAct loops, and what does it imply for approach choice?"
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["report"].strip(), "the team must produce a report"
        # The delegation is the whole point — assert it actually happened.
        assert body["subagents_used"], "lead must delegate via the Task tool"
        assert set(body["subagents_used"]) <= {"researcher", "analyst", "writer"}
        assert "Task" in body["tools_used"]
