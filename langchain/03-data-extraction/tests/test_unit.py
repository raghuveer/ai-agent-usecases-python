# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langchain). See langchain/03-data-extraction/README.md
"""Unit tests for UC3 data-extraction (langchain) — fully mocked, no network.

We inject ``FakeListChatModel`` onto ``app.state`` and exercise the extraction
chain directly with the same fakes.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app import extract
from app.extract import (
    ExtractionError,
    Invoice,
    extract_invoice,
    parse_and_validate,
)
from app.main import app

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
    app.state.llm = FakeListChatModel(responses=responses)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Schema + JSON extraction
# --------------------------------------------------------------------------- #
def test_sample_invoice_fixture_exists():
    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    assert "INV-2045" in text


def test_extract_json_object_strips_fences_and_prose():
    wrapped = "Sure:\n```json\n" + VALID_JSON + "\n```\nDone."
    inv = parse_and_validate(wrapped)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22


def test_system_prompt_includes_schema_placeholder():
    assert "invoice_number" in extract.INVOICE_SCHEMA_JSON
    assert "{schema}" in extract.SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# extract_invoice: happy path + retry path (no network)
# --------------------------------------------------------------------------- #
def test_extract_invoice_valid_first_try():
    llm = FakeListChatModel(responses=[VALID_JSON])
    inv = extract_invoice("raw invoice text", llm, "qwen-local-instruct")
    assert isinstance(inv, Invoice)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22


def test_extract_invoice_retries_once_then_succeeds():
    # First reply invalid, second valid -> retry path succeeds.
    llm = FakeListChatModel(responses=[INVALID_JSON, VALID_JSON])
    inv = extract_invoice("raw invoice text", llm, "qwen-local-instruct")
    assert inv.total == 1903.22


def test_extract_invoice_raises_after_retry_still_invalid():
    llm = FakeListChatModel(responses=[INVALID_JSON, INVALID_JSON])
    with pytest.raises(ExtractionError):
        extract_invoice("raw invoice text", llm, "qwen-local-instruct")


def test_extract_invoice_text_mode_explicit():
    # Passing structured_mode="text" must behave like the default text path.
    llm = FakeListChatModel(responses=[VALID_JSON])
    inv = extract_invoice(
        "raw invoice text", llm, "qwen-local-instruct", structured_mode="text"
    )
    assert isinstance(inv, Invoice)
    assert inv.invoice_number == "INV-2045"


# --------------------------------------------------------------------------- #
# native structured-output mode (mocked with_structured_output -> Invoice)
# --------------------------------------------------------------------------- #
class FakeStructuredModel(FakeListChatModel):
    """Fake whose ``with_structured_output`` returns a Runnable yielding objects."""

    structured_returns: list = []

    def with_structured_output(self, schema, **kwargs):  # noqa: D401, ARG002
        outputs = list(self.structured_returns)

        def _emit(_inputs):
            value = outputs.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        return RunnableLambda(_emit)


def _native_model(returns: list) -> FakeStructuredModel:
    m = FakeStructuredModel(responses=["unused"])
    m.structured_returns = returns
    return m


VALID_INVOICE = Invoice(
    invoice_number="INV-2045",
    vendor="Northwind Robotics",
    date="2026-05-14",
    total=1903.22,
    line_items=[{"description": "ServoMax 9000 actuator", "amount": 1250.00}],
)


def test_extract_invoice_native_first_try():
    llm = _native_model([VALID_INVOICE])
    inv = extract_invoice(
        "raw invoice text", llm, "claude-haiku-4-5", structured_mode="native"
    )
    assert isinstance(inv, Invoice)
    assert inv.invoice_number == "INV-2045"
    assert inv.total == 1903.22


def test_extract_invoice_native_retries_once_then_succeeds():
    llm = _native_model([ValueError("bad output"), VALID_INVOICE])
    inv = extract_invoice(
        "raw invoice text", llm, "claude-haiku-4-5", structured_mode="native"
    )
    assert inv.total == 1903.22


def test_extract_invoice_native_raises_after_retry():
    llm = _native_model([ValueError("bad"), ValueError("still bad")])
    with pytest.raises(ExtractionError):
        extract_invoice(
            "raw invoice text", llm, "claude-haiku-4-5", structured_mode="native"
        )


# --------------------------------------------------------------------------- #
# HTTP level (fake LLM, no network)
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
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


def test_strip_thinking_removes_qwen3_think_blocks():
    """`/no_think` still emits an empty <think></think> pair; it must not ship.

    Found by running the Docker quickstart, whose default model is a qwen3 tag:
    answers came back with a leading empty thinking block before the text.
    """
    from app.llm import strip_thinking

    empty_block = "<think>" + "\n\n" + "</think>" + "\n\n" + "30 days."
    assert strip_thinking(empty_block) == "30 days."
    assert strip_thinking("<think>reasoning</think>Answer.") == "Answer."
    assert strip_thinking("  30 days.  ") == "30 days."
    # Chain-of-thought must never survive into a response.
    assert "reasoning" not in strip_thinking("<think>reasoning</think>ok")
