# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC05 Customer support triage (claude-agent-sdk). See claude-agent-sdk/05-support-triage/README.md
"""Unit tests for UC05 support-triage (claude-agent-sdk). Stubbed agent, no network.

Focus: the decision contract (enums must be enforced, not merely suggested) and
the auditability of the agent's routing choice — did it actually look the order
up, or answer from the ticket text alone?
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.settings import Settings
from app.triage import EMIT_TOOL, LOOKUP_TOOL, lookup_order, triage

DECISION = {
    "category": "shipping",
    "priority": "urgent",
    "needs_human": True,
    "reply": "Your parcel is marked lost in transit; we have opened a case and will follow up today.",
}


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text="Triage recorded.",
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name=LOOKUP_TOOL, input={"order_id": "A-1003"}),
                ToolCall(name=EMIT_TOOL, input=dict(DECISION)),
            ],
            num_turns=3,
            cost_usd=0.002,
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
        "usecase": "05-support-triage",
    }


def test_run_returns_validated_decision_and_lookup_trail():
    body = make_client().post("/run", json={"ticket": "Where is order A-1003?"}).json()
    assert body["valid"] is True
    assert body["decision"]["category"] == "shipping"
    assert body["decision"]["priority"] == "urgent"
    assert body["decision"]["needs_human"] is True
    assert body["order_lookups"] == ["A-1003"]


def test_agent_may_skip_the_lookup_and_that_is_visible():
    """Routing is the agent's call — the response records what it actually did."""
    calls = [ToolCall(name=EMIT_TOOL, input=dict(DECISION, category="billing"))]
    body = make_client(tool_calls=calls).post("/run", json={"ticket": "invoice?"}).json()
    assert body["valid"] is True
    assert body["order_lookups"] == []


def test_missing_decision_is_reported():
    body = make_client(tool_calls=[]).post("/run", json={"ticket": "x"}).json()
    assert body["valid"] is False
    assert body["decision"] is None
    assert "did not call emit_triage" in body["errors"][0]


@pytest.mark.parametrize(
    "bad,field",
    [
        ({"category": "refunds"}, "category"),
        ({"priority": "critical"}, "priority"),
        ({"needs_human": "maybe"}, "needs_human"),
        ({"reply": "x" * 2001}, "reply"),
    ],
)
def test_out_of_contract_values_are_rejected(bad, field):
    """The enums are a contract; an off-list value must not pass through."""
    payload = dict(DECISION)
    payload.update(bad)
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=payload)])
        .post("/run", json={"ticket": "x"})
        .json()
    )
    assert body["valid"] is False
    assert any(field in e for e in body["errors"])


@pytest.mark.parametrize("raw,expected", [("yes", True), ("true", True), ("no", False)])
def test_boolean_like_strings_are_coerced_not_rejected(raw, expected):
    """Documented behaviour: Pydantic's lax mode accepts yes/no/true/false.

    That is deliberate here — models routinely emit `"true"` rather than `true`,
    and coercing is friendlier than failing. Genuinely ambiguous values like
    "maybe" still fail (see the test above).
    """
    payload = dict(DECISION, needs_human=raw)
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=payload)])
        .post("/run", json={"ticket": "x"})
        .json()
    )
    assert body["valid"] is True
    assert body["decision"]["needs_human"] is expected


def test_last_emit_wins():
    calls = [
        ToolCall(name=EMIT_TOOL, input=dict(DECISION, category="other")),
        ToolCall(name=EMIT_TOOL, input=dict(DECISION, category="billing")),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"ticket": "x"}).json()
    assert body["decision"]["category"] == "billing"


def test_schema_and_orders_endpoints():
    client = make_client()
    schema = client.get("/schema").json()
    assert set(schema["required"]) == {"category", "priority", "needs_human", "reply"}
    assert "A-1003" in client.get("/orders").json()


@pytest.mark.anyio
async def test_lookup_order_returns_details():
    out = await lookup_order.handler({"order_id": "a-1002"})  # case-insensitive
    assert "in_transit" in out["content"][0]["text"]


@pytest.mark.anyio
async def test_lookup_order_unknown_is_recoverable_error():
    out = await lookup_order.handler({"order_id": "ZZZ"})
    assert out["is_error"] is True


@pytest.mark.anyio
async def test_options_register_both_tools():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        return AgentResult(text="", tool_calls=[ToolCall(name=EMIT_TOOL, input=dict(DECISION))])

    await triage("t", Settings(), spy)
    assert seen["tools"] == [LOOKUP_TOOL, EMIT_TOOL]


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"ticket": "where is my order?"}


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
