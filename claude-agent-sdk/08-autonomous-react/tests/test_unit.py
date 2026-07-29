# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC08 Autonomous ReAct (claude-agent-sdk). See claude-agent-sdk/08-autonomous-react/README.md
"""Unit tests for UC08 autonomous-react (claude-agent-sdk). Stubbed agent, no network.

The tools are real and are tested directly — they are the only substantive logic
here, since the SDK supplies the loop. `safe_eval` gets adversarial cases because
it evaluates model-authored text.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.react_agent import METRICS, calculate, lookup_metric, run_react, safe_eval
from app.settings import Settings


def make_runner(tool_calls=None, stop_reason=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text="Margin is 28.9%.",
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(
                    name="mcp__metrics__lookup_metric",
                    input={"name": "monthly_revenue_usd"},
                ),
                ToolCall(
                    name="mcp__metrics__calculate",
                    input={"expression": "(128400 - 91250) / 128400"},
                ),
            ],
            num_turns=4,
            cost_usd=0.004,
            stop_reason=stop_reason,
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
        "usecase": "08-autonomous-react",
    }


def test_run_returns_answer_and_readable_trace():
    body = make_client().post("/run", json={"question": "What is our margin?"}).json()
    assert body["answer"] == "Margin is 28.9%."
    # mcp__server__ prefixes are stripped so the trace reads as tool names.
    assert [s["tool"] for s in body["trace"]] == ["lookup_metric", "calculate"]
    assert body["trace"][0]["input"] == {"name": "monthly_revenue_usd"}
    assert body["hit_turn_limit"] is False


def test_turn_limit_is_reported_not_hidden():
    body = make_client(stop_reason="max_turns").post("/run", json={"question": "q"}).json()
    assert body["hit_turn_limit"] is True


def test_metrics_endpoint_exposes_the_warehouse():
    assert make_client().get("/metrics").json() == METRICS


# --- tools -------------------------------------------------------------------


@pytest.mark.anyio
async def test_lookup_metric_returns_value():
    out = await lookup_metric.handler({"name": "active_customers"})
    assert "1820" in out["content"][0]["text"].replace(".0", "")


@pytest.mark.anyio
async def test_lookup_metric_unknown_is_a_recoverable_error():
    out = await lookup_metric.handler({"name": "nope"})
    assert out["is_error"] is True
    # The error lists valid names so the agent can correct itself.
    assert "monthly_revenue_usd" in out["content"][0]["text"]


@pytest.mark.anyio
async def test_calculate_evaluates_arithmetic():
    out = await calculate.handler({"expression": "(128400 - 91250) / 128400"})
    assert out["content"][0]["text"].startswith("0.289")


@pytest.mark.anyio
async def test_calculate_returns_error_result_instead_of_raising():
    out = await calculate.handler({"expression": "1/0"})
    assert out["is_error"] is True
    assert "Error" in out["content"][0]["text"]


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__.__mro__[1].__subclasses__()",
        "1 if True else 2",
        "'a' * 99",
        "x + 1",
    ],
)
def test_safe_eval_rejects_non_arithmetic(expr):
    """It evaluates model-authored text, so anything but arithmetic must fail."""
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_safe_eval_rejects_overlong_input():
    with pytest.raises(ValueError):
        safe_eval("1+" * 200 + "1")


def test_safe_eval_handles_valid_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14.0
    assert safe_eval("-(10 / 4)") == -2.5
    assert safe_eval("2 ** 10") == 1024.0


@pytest.mark.anyio
async def test_options_register_tools_and_raise_turn_ceiling():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        seen["servers"] = list(options.mcp_servers)
        seen["max_turns"] = options.max_turns
        return AgentResult(text="ok")

    await run_react("q", Settings(), spy)
    assert seen["servers"] == ["metrics"]
    assert seen["tools"] == [
        "mcp__metrics__lookup_metric",
        "mcp__metrics__calculate",
    ]
    assert seen["max_turns"] >= 8
