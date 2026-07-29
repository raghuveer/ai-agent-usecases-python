# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC02 Code generation (claude-agent-sdk). See claude-agent-sdk/02-code-generation/README.md
"""Integration test — a real agent writes code, writes tests, and runs them.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1) and capped by
AGENT_MAX_TURNS / AGENT_MAX_BUDGET_USD. Needs Node.js + the Claude Code CLI on
PATH, since the Python SDK spawns it.
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
def test_agent_writes_and_runs_its_own_tests():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "task": "Write a function `fizzbuzz(n)` returning a list of "
                "strings for 1..n with the usual Fizz/Buzz rules."
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["solution"].strip(), "the agent must produce solution.py"
        assert body["tests"].strip(), "the agent must produce test_solution.py"
        assert "fizzbuzz" in body["solution"].lower()
        # The whole point of this approach: it executed the tests itself.
        assert "Bash" in body["tools_used"], body["tools_used"]
        assert body["tests_passed"] is True, body["summary"]
