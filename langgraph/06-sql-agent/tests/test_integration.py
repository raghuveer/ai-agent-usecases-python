"""Integration test for UC6 sql-agent (langgraph).

Hits the live local Qwen coder via the gateway and a real SQLite DB built from
the bundled seed. Skipped unless RUN_INTEGRATION=1.
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
    with TestClient(app) as client:  # fires startup -> build DB + compile graph
        resp = client.post("/run", json={"question": "How many customers are there?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"].upper().startswith("SELECT")
    assert "COUNT" in body["sql"].upper()
    assert body["rows"]
    only_value = list(body["rows"][0].values())[0]
    assert int(only_value) == 5
