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
from app.extract import EMIT_TOOL, extract
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
