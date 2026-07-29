# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC08 Autonomous ReAct (claude-agent-sdk). See claude-agent-sdk/08-autonomous-react/README.md
"""Integration test — a real agent chooses and sequences tools on its own.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1), capped by
AGENT_MAX_TURNS / AGENT_MAX_BUDGET_USD. Needs Node.js + the Claude Code CLI.

The assertion that matters: the agent must actually *call the tools*. Answering
from the prompt alone would mean it invented the numbers.
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
def test_agent_uses_tools_to_reach_a_multi_step_answer():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "question": "What is our gross margin as a percentage of monthly "
                "revenue? Use the metrics tools."
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        tools_used = [step["tool"] for step in body["trace"]]
        assert "lookup_metric" in tools_used, f"agent must fetch metrics: {tools_used}"
        # Multi-step: it needed more than one tool call to get there.
        assert len(body["trace"]) >= 2, body["trace"]
        assert body["answer"].strip()
        assert body["hit_turn_limit"] is False, "raise AGENT_MAX_TURNS if this trips"
