# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langgraph). See langgraph/05-support-triage/README.md
"""Integration test for UC5 support-triage (langgraph).

Hits the live local Qwen via the gateway, driving the compiled triage graph.
Skipped unless RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app import triage
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run against the live local model",
)
def test_billing_message_classifies_against_local_qwen():
    triage.reset_memory()
    app = create_app()
    with TestClient(app) as client:  # fires startup -> build graph
        resp = client.post(
            "/run",
            json={"message": "I was double charged on my invoice", "session_id": "it-1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "billing"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["response"].strip()
