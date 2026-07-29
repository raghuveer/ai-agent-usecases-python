# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC03 Data extraction (claude-agent-sdk). See claude-agent-sdk/03-data-extraction/README.md
"""Integration test — a real agent extracts an invoice via the emit tool.

Double-gated (RUN_INTEGRATION=1 + RUN_ANTHROPIC_TESTS=1). Needs Node.js + the
Claude Code CLI.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1" and os.getenv("RUN_ANTHROPIC_TESTS") == "1"
REASON = "set RUN_INTEGRATION=1 and RUN_ANTHROPIC_TESTS=1 to run live agent tests"

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample-invoice.txt"


@pytest.mark.skipif(not RUN, reason=REASON)
def test_extracts_sample_invoice_into_valid_record():
    document = SAMPLE.read_text(encoding="utf-8")
    with TestClient(create_app()) as client:
        resp = client.post("/run", json={"document": document})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["valid"] is True, body["errors"]
        invoice = body["invoice"]
        assert invoice["invoice_number"] == "INV-2026-0042"
        assert "Northwind" in invoice["vendor"]
        assert invoice["currency"].upper() == "USD"
        # The number must survive the "$1,284.50" formatting.
        assert invoice["total"] == pytest.approx(1284.50)
