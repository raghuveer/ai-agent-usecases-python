# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC09 Personalised recommendations (claude-agent-sdk). See claude-agent-sdk/09-recommendations/README.md
"""Integration test — a real agent fetches profile + catalog and ranks.

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
def test_recommendations_respect_the_profile():
    with TestClient(create_app()) as client:
        resp = client.post("/run", json={"user_id": "u-1"})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["valid"] is True, body["errors"]
        assert body["items"], "expected at least one recommendation"

        categories = {i["category"] for i in body["items"]}
        names = {i["name"] for i in body["items"]}
        # u-1 dislikes kitchen and just bought a Mechanical Keyboard.
        assert "kitchen" not in categories, body["items"]
        assert "Mechanical Keyboard" not in names, body["items"]
        # Budget is $450/item.
        assert all(i["price"] <= 450 for i in body["items"]), body["items"]
        assert all(i["reason"].strip() for i in body["items"])
