# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langgraph). See langgraph/03-data-extraction/README.md
"""Unit tests for UC3 data-extraction (langgraph). Fake LLM, no network."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import extract
from app.extract import (
    ExtractionError,
    Invoice,
    build_extract_graph,
    extract_invoice,
    parse_and_validate,
)
from app.main import create_app

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


def make_client(responses: list[str]) -> TestClient:
    llm = FakeListChatModel(responses=responses)
    return TestClient(create_app(llm=llm))


# --------------------------------------------------------------------------- #
# Schema + JSON extraction
# --------------------------------------------------------------------------- #
def test_sample_invoice_fixture_exists():
    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    assert "INV-2045" in text


def test_extract_json_object_strips_fences_and_prose():
    wrapped = "Result:\n```json\n" + VALID_JSON + "\n```"
    inv = parse_and_validate(wrapped)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22


def test_system_instruction_includes_schema():
    assert "invoice_number" in extract.SYSTEM_INSTRUCTION
    assert "line_items" in extract.SYSTEM_INSTRUCTION


# --------------------------------------------------------------------------- #
# Graph: happy path + retry path (no network)
# --------------------------------------------------------------------------- #
def test_extract_invoice_valid_first_try():
    llm = FakeListChatModel(responses=[VALID_JSON])
    inv = extract_invoice("raw invoice text", llm)
    assert isinstance(inv, Invoice)
    assert inv.total == 1903.22


def test_graph_retries_once_then_succeeds():
    # First reply invalid -> graph loops back to extract -> second reply valid.
    llm = FakeListChatModel(responses=[INVALID_JSON, VALID_JSON])
    graph = build_extract_graph(llm)
    out = graph.invoke(
        {"text": "t", "attempts": 0, "raw": "", "error": "", "invoice": None}
    )
    assert out["invoice"] is not None
    assert out["invoice"].total == 1903.22
    assert out["attempts"] == 2  # exactly one retry


def test_extract_invoice_raises_after_retry_still_invalid():
    llm = FakeListChatModel(responses=[INVALID_JSON, INVALID_JSON])
    with pytest.raises(ExtractionError):
        extract_invoice("raw invoice text", llm)


# --------------------------------------------------------------------------- #
# HTTP level (fake LLM, no network)
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "03-data-extraction",
    }


def test_run_extracts_and_validates():
    client = make_client([VALID_JSON])
    resp = client.post("/run", json={"text": "some invoice text"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["invoice_number"] == "INV-2045"
    assert body["total"] == 1903.22
    assert len(body["line_items"]) == 2


def test_run_returns_422_when_invalid_after_retry():
    client = make_client([INVALID_JSON, INVALID_JSON])
    resp = client.post("/run", json={"text": "bad invoice"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["valid"] is False
