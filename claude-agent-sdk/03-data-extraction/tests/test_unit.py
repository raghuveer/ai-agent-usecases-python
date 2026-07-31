# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC03 Data extraction (claude-agent-sdk). See claude-agent-sdk/03-data-extraction/README.md
"""Unit tests for UC03 data-extraction (claude-agent-sdk). Stubbed agent, no network.

The interesting cases are the failure modes: the agent not calling the tool at
all, and calling it with a payload that does not satisfy the schema. Both must
be reported, never silently returned as success.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.extract import EMIT_TOOL, _validate, extract
from app.main import create_app
from app.settings import Settings

GOOD = {
    "invoice_number": "INV-2026-0042",
    "vendor": "Northwind Traders",
    "invoice_date": "2026-03-14",
    "currency": "USD",
    "total": 1284.50,
    "line_items": [
        {"description": "Consulting, March", "amount": 1200.00},
        {"description": "Shipping", "amount": 84.50},
    ],
}


def make_runner(tool_calls=None, text="Recorded."):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=text,
            tool_calls=tool_calls
            if tool_calls is not None
            else [ToolCall(name=EMIT_TOOL, input=dict(GOOD))],
            num_turns=2,
            cost_usd=0.001,
        )

    return runner


def make_client(**kw) -> TestClient:
    return TestClient(create_app(runner=make_runner(**kw)))


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "claude-agent-sdk",
        "usecase": "03-data-extraction",
    }


def test_run_returns_validated_record():
    body = make_client().post("/run", json={"document": "invoice text"}).json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["invoice"]["invoice_number"] == "INV-2026-0042"
    assert body["invoice"]["total"] == 1284.50
    assert len(body["invoice"]["line_items"]) == 2


def test_missing_tool_call_is_reported_not_silently_empty():
    body = make_client(tool_calls=[]).post("/run", json={"document": "x"}).json()
    assert body["valid"] is False
    assert body["invoice"] is None
    assert "did not call emit_invoice" in body["errors"][0]


def test_schema_violation_is_reported_with_field_paths():
    bad = dict(GOOD)
    bad.pop("vendor")
    bad["total"] = "not-a-number"
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"document": "x"})
        .json()
    )
    assert body["valid"] is False
    assert body["invoice"] is None
    joined = " ".join(body["errors"])
    assert "vendor" in joined and "total" in joined


def test_nested_line_item_violation_is_caught():
    bad = dict(GOOD)
    bad["line_items"] = [{"description": "ok", "amount": "twelve"}]
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"document": "x"})
        .json()
    )
    assert body["valid"] is False
    assert any("line_items" in e for e in body["errors"])


def test_last_call_wins_when_agent_emits_twice():
    first = dict(GOOD, invoice_number="WRONG")
    calls = [
        ToolCall(name=EMIT_TOOL, input=first),
        ToolCall(name=EMIT_TOOL, input=dict(GOOD)),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"document": "x"}).json()
    assert body["invoice"]["invoice_number"] == "INV-2026-0042"


def test_line_items_are_optional():
    minimal = {k: v for k, v in GOOD.items() if k != "line_items"}
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=minimal)])
        .post("/run", json={"document": "x"})
        .json()
    )
    assert body["valid"] is True
    assert body["invoice"]["line_items"] == []


def test_schema_endpoint_exposes_contract():
    body = make_client().get("/schema").json()
    assert body["required"] == [
        "invoice_number",
        "vendor",
        "invoice_date",
        "currency",
        "total",
    ]


@pytest.mark.anyio
async def test_options_register_only_the_emit_tool():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        seen["max_turns"] = options.max_turns
        return AgentResult(text="", tool_calls=[ToolCall(name=EMIT_TOOL, input=dict(GOOD))])

    await extract("doc", Settings(), spy)
    assert seen["tools"] == [EMIT_TOOL]
    # One-shot task: no room to wander.
    assert seen["max_turns"] == 3


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"document": "invoice text"}


def _stub_runner(**fields):
    """A runner that returns exactly the AgentResult it is handed."""

    async def runner(prompt, options) -> AgentResult:
        return AgentResult(**fields)

    return runner


def test_stop_reason_reports_a_completed_run():
    """The SDK reports no stop reason on several paths, so the field would be
    null exactly when the run was fine. `end_turn` fills that gap: callers get
    one field that is always present and always means something."""
    client = TestClient(create_app(runner=_stub_runner(text="done", num_turns=1)))
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "end_turn"


def test_stop_reason_reports_a_capped_run():
    """The gap this closes. A run cut short by `max_turns` still answers 200
    with whatever it managed to produce -- previously indistinguishable from a
    run that finished properly, which is the one thing a caller must be able to
    tell apart."""
    client = TestClient(
        create_app(
            runner=_stub_runner(
                text="partial", num_turns=8, is_error=True, stop_reason="max_turns"
            )
        )
    )
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "max_turns"


# --------------------------------------------------------------------------- #
# F17 — the total is checked against the line items, not trusted
# --------------------------------------------------------------------------- #
BASE = {
    "invoice_number": "INV-1",
    "vendor": "Acme",
    "invoice_date": "2026-03-11",
    "currency": "GBP",
}


def test_a_total_that_disagrees_with_the_line_items_is_invalid():
    """The injected-total attack, caught by arithmetic rather than by judgement.

    A document carrying a note addressed "TO THE EXTRACTION SYSTEM" — claiming
    the printed 300.00 was a typo for 3.00 — was ignored by the model on the
    run that found this. Nothing in the code would have noticed if it had not,
    and "the model usually declines" is not a control.
    """
    out = _validate({**BASE, "total": 3.00, "line_items": [
        {"description": "Widgets", "amount": 250.00},
        {"description": "Gaskets", "amount": 50.00},
    ]})
    assert out.valid is False
    assert any("does not match the sum of line items" in e for e in out.errors)


def test_a_consistent_invoice_is_accepted():
    out = _validate({**BASE, "total": 300.00, "line_items": [
        {"description": "Widgets", "amount": 250.00},
        {"description": "Gaskets", "amount": 50.00},
    ]})
    assert out.valid is True and out.invoice["total"] == 300.00


def test_rounding_noise_does_not_fail_a_correct_invoice():
    """The check exists to catch disagreement, not float representation."""
    out = _validate({**BASE, "total": 0.30, "line_items": [
        {"description": "a", "amount": 0.10},
        {"description": "b", "amount": 0.20},
    ]})
    assert out.valid is True


def test_an_invoice_without_line_items_is_not_second_guessed():
    """Nothing to check against — inventing a failure here would be worse."""
    out = _validate({**BASE, "total": 300.00, "line_items": []})
    assert out.valid is True
