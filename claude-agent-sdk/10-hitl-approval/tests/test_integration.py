# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (claude-agent-sdk). See claude-agent-sdk/10-hitl-approval/README.md
"""Integration test — real agent loop parks at the gate, then resumes.

Double-gated: needs RUN_INTEGRATION=1 **and** RUN_ANTHROPIC_TESTS=1. Unlike the
other three approaches, this one cannot fall back to free local Qwen — the Agent
SDK drives the Claude Code harness, which small local models cannot sustain (see
README). So every live run here spends a small, capped amount of Anthropic
budget: `AGENT_MAX_TURNS` and `AGENT_MAX_BUDGET_USD` bound it.

Also requires Node.js and the Claude Code CLI on PATH — the Python SDK spawns it.
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
def test_run_parks_then_resume_executes():
    # Context-managed so one event loop spans both requests and the parked
    # agent coroutine survives between them.
    with TestClient(create_app()) as client:
        run = client.post(
            "/run",
            json={
                "request": "Tell customer@example.com their $40 refund for a "
                "defective robot vacuum is approved."
            },
        )
        assert run.status_code == 200, run.text
        body = run.json()
        assert body["status"] == "awaiting_approval", body
        assert body["proposed_action"].strip(), "the gate must surface a draft"
        run_id = body["run_id"]

        resumed = client.post("/resume", json={"run_id": run_id, "approved": True})
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "executed"


@pytest.mark.skipif(not RUN, reason=REASON)
def test_rejection_stops_the_run():
    with TestClient(create_app()) as client:
        run_id = client.post(
            "/run",
            json={"request": "Tell customer@example.com we are closing their account."},
        ).json()["run_id"]

        resumed = client.post(
            "/resume",
            json={"run_id": run_id, "approved": False, "feedback": "Not authorised."},
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "rejected"
