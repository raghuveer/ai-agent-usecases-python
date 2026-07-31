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

import importlib

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
    assert [s["tool"] for s in body["steps"]] == ["lookup_metric", "calculate"]
    assert body["steps"][0]["input"] == {"name": "monthly_revenue_usd"}
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


# --------------------------------------------------------------------------- #
# Tracing — deliberately partial. See app/trace.py and docs/trace-format.md
# --------------------------------------------------------------------------- #
def test_trace_absent_unless_requested():
    assert make_client().post("/run", json={"question": "x"}).json()["trace"] is None


def test_trace_reports_tool_calls_and_real_cost():
    trace = make_client().post(
        "/run?trace=1", json={"question": "x"}
    ).json()["trace"]

    assert trace["schema_version"] == 1
    assert trace["approach"] == "claude-agent-sdk"
    # This approach talks the Anthropic protocol, not the OpenAI one.
    assert trace["gen_ai"]["system"] == "anthropic"
    assert [(s["type"], s["name"]) for s in trace["spans"]] == [
        ("tool", "lookup_metric"),
        ("tool", "calculate"),
    ]
    # The one number this approach knows better than the other three.
    assert trace["outcome"]["cost_usd"] == 0.004
    assert trace["outcome"]["steps"] == 4  # turns, not model calls


def test_trace_marks_what_the_sdk_cannot_expose_as_absent_not_zero():
    """The asymmetry is the finding: no loop of your own, no visibility.

    A zero token count would read as a measurement ("this run was free"), so the
    unknowable fields are null and listed explicitly in `not_captured`.
    """
    trace = make_client().post(
        "/run?trace=1", json={"question": "x"}
    ).json()["trace"]

    assert trace["gen_ai"]["usage"] == {"input_tokens": None, "output_tokens": None}
    for span in trace["spans"]:
        assert span["response"] is None      # the harness kept the tool result
        assert span["duration_ms"] is None   # per-call latency is unobservable
        assert "messages" not in span.get("request", {})

    assert len(trace["not_captured"]) == 4


def test_capped_run_is_traced_as_capped():
    trace = make_client(stop_reason="max_turns").post(
        "/run?trace=1", json={"question": "x"}
    ).json()["trace"]
    assert trace["outcome"]["status"] == "capped"
    assert trace["outcome"]["stop_reason"] == "max_turns"


# --------------------------------------------------------------------------- #
# Streaming (SSE) — turns, not tokens. See docs/streaming.md
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_iter_events_yields_turns_and_steps_but_never_tokens():
    """The asymmetry, asserted: the SDK exposes no deltas, so there is no
    `token` frame to emit. Writing no loop costs you the view inside it."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

    from app import agent as agent_mod

    async def fake_query(**_kwargs):
        yield AssistantMessage(
            content=[TextBlock(text="Looking that up.")], model="m"
        )
        yield AssistantMessage(
            content=[ToolUseBlock(id="t1", name="mcp__metrics__lookup_metric",
                                  input={"name": "revenue"})],
            model="m",
        )

    agent_mod.query = fake_query
    try:
        events = [
            e async for e in agent_mod.iter_events(
                "q", agent_mod.ClaudeAgentOptions()
            )
        ]
    finally:
        importlib.reload(agent_mod)

    kinds = [e["type"] for e in events]
    assert "token" not in kinds, "the SDK cannot stream tokens"
    assert kinds == ["turn", "step", "final"]
    assert events[1]["step"]["tool"] == "lookup_metric"


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"question": "q"}


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
