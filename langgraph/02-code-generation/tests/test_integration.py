# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langgraph). See langgraph/02-code-generation/README.md
"""Integration test for UC2 code-generation (langgraph).

Hits the live local Qwen coder via the gateway. Skipped unless RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run against the live local model",
)
def test_run_against_local_qwen():
    app = create_app()
    with TestClient(app) as client:  # fires startup -> build llm + graph
        resp = client.post(
            "/run",
            json={
                "task": "Write a Python function add(a,b) that returns their sum",
                "language": "python",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "python"
    assert body["code"].strip()
    assert "def add" in body["code"]
