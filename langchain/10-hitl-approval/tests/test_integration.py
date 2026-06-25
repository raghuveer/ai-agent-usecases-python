"""Integration test — full /run -> /resume(approved=true) cycle vs local Qwen.

Gated: runs only when RUN_INTEGRATION=1, otherwise skipped. Uses the free local
model via the gateway (no Anthropic budget).
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_run_then_resume_cycle_against_local_qwen():
    # Context-manage so the lifespan builds the real LLM chain.
    with TestClient(create_app()) as client:
        run = client.post(
            "/run",
            json={"request": "Approve a $40 refund for a defective robot vacuum."},
        )
        assert run.status_code == 200
        body = run.json()
        assert body["status"] == "awaiting_approval"
        assert body["proposed_action"].strip(), "proposed_action must be non-empty"
        run_id = body["run_id"]

        resumed = client.post("/resume", json={"run_id": run_id, "approved": True})
        assert resumed.status_code == 200
        rbody = resumed.json()
        assert rbody["status"] == "executed"
        assert rbody["result"].strip()
