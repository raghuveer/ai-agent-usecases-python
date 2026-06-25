# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (raw-api). See raw-api/03-data-extraction/README.md
"""Unit tests for UC3 data-extraction (raw-api) — fully mocked, no network."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import extract
from app.extract import ExtractionError, Invoice, extract_invoice, parse_and_validate
from app.main import create_app
from app.settings import Settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VALID_JSON = """{
  "invoice_number": "INV-2045",
  "vendor": "Northwind Robotics",
  "date": "2026-05-14",
  "total": 1903.22,
  "line_items": [
    {"description": "ServoMax 9000 actuator", "amount": 1250.00},
    {"description": "Lidar sensor module", "amount": 480.50}
  ]
}"""

# Missing required `total` -> Pydantic validation error.
INVALID_JSON = """{
  "invoice_number": "INV-2045",
  "vendor": "Northwind Robotics",
  "date": "2026-05-14",
  "line_items": []
}"""


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


class RecordingLLM:
    """Returns canned replies in order; records every call it receives."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._replies.pop(0)


# --------------------------------------------------------------------------- #
# Schema + JSON extraction
# --------------------------------------------------------------------------- #
def test_sample_invoice_fixture_exists():
    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    assert "INV-2045" in text


def test_invoice_schema_shape():
    inv = Invoice.model_validate(
        {
            "invoice_number": "X",
            "vendor": "V",
            "date": "2026-01-01",
            "total": 1.0,
            "line_items": [{"description": "d", "amount": 2.0}],
        }
    )
    assert inv.total == 1.0
    assert inv.line_items[0].amount == 2.0


def test_extract_json_object_strips_fences_and_prose():
    wrapped = "Here you go:\n```json\n" + VALID_JSON + "\n```\nThanks!"
    inv = parse_and_validate(wrapped)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22


# --------------------------------------------------------------------------- #
# extract_invoice: happy path + retry path (no network)
# --------------------------------------------------------------------------- #
def test_extract_invoice_valid_first_try():
    llm = RecordingLLM([VALID_JSON])
    inv = extract_invoice("raw invoice text", llm_call=llm)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22
    assert len(llm.calls) == 1  # no retry needed


def test_extract_invoice_retries_once_then_succeeds():
    llm = RecordingLLM([INVALID_JSON, VALID_JSON])
    inv = extract_invoice("raw invoice text", llm_call=llm)
    assert inv.total == 1903.22
    # Exactly two calls: first failed validation, retry succeeded.
    assert len(llm.calls) == 2
    # The retry prompt carried the validation error back to the model.
    assert "could not be validated" in llm.calls[1][1]
    assert "Error:" in llm.calls[1][1]


def test_extract_invoice_raises_after_retry_still_invalid():
    llm = RecordingLLM([INVALID_JSON, INVALID_JSON])
    with pytest.raises(ExtractionError):
        extract_invoice("raw invoice text", llm_call=llm)
    assert len(llm.calls) == 2  # one retry, no more


def test_system_prompt_includes_schema():
    assert "invoice_number" in extract.SYSTEM_PROMPT
    assert "line_items" in extract.SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(VALID_JSON)

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "03-data-extraction",
    }

    r = client.post("/run", json={"text": "some invoice text"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["invoice_number"] == "INV-2045"
    assert body["total"] == 1903.22
    assert len(body["line_items"]) == 2


def test_run_returns_422_when_invalid_after_retry():
    app = create_app()
    app.state.settings = Settings()
    # Always invalid -> first try + retry both fail -> 422.
    app.state.client = _fake_openai_client(INVALID_JSON)

    client = TestClient(app)
    r = client.post("/run", json={"text": "bad invoice"})
    assert r.status_code == 422
    assert r.json()["detail"]["valid"] is False
