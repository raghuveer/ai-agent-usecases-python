# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC05 Customer support triage (claude-agent-sdk). See claude-agent-sdk/05-support-triage/README.md
"""Integration test — a real agent triages tickets and chooses whether to enrich.

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
def test_lost_parcel_is_looked_up_and_escalated():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={
                "ticket": "Order A-1003 was due last week and still hasn't arrived. "
                "I need it for a birthday on Saturday."
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["valid"] is True, body["errors"]
        # The order id is in the ticket and status changes the answer — expect a lookup.
        assert "A-1003" in body["order_lookups"], body["order_lookups"]
        assert body["decision"]["category"] == "shipping"
        assert body["decision"]["priority"] in ("high", "urgent")


@pytest.mark.skipif(not RUN, reason=REASON)
def test_generic_question_needs_no_order_lookup():
    with TestClient(create_app()) as client:
        resp = client.post(
            "/run",
            json={"ticket": "Do you ship to Ireland, and how long does it take?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["valid"] is True, body["errors"]
        # No order id mentioned — a lookup here would be wasted work.
        assert body["order_lookups"] == [], body["order_lookups"]
