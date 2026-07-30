# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langchain). See langchain/08-autonomous-react/README.md
"""Unit tests for UC8 autonomous-react (langchain) — fully mocked, no network.

We drive the agent with ``FakeListChatModel`` scripted to force BOTH tools,
exercise the unsafe-calculator rejection, and cover both ``stopped_reason``
outcomes.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import llm as llm_mod
from app import react, tools
from app.main import app
from app.tools import build_tools


# --------------------------------------------------------------------------- #
# tools: calculator safety + search
# --------------------------------------------------------------------------- #
def test_calculator_evaluates_arithmetic():
    assert tools.calculator("30 * 2") == "60"
    assert tools.calculator("(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_input():
    import pytest

    for hostile in ["__import__('os')", "open('x')", "a + b", "1; 2"]:
        with pytest.raises(tools.UnsafeExpression):
            tools.safe_eval(hostile)


def test_calculator_tool_surfaces_unsafe_as_error():
    tmap = {t.name: t for t in build_tools()}
    out = tmap["calculator"].invoke("__import__('os')")
    assert "unsafe" in out.lower()


def test_search_finds_fact():
    assert "30 days" in tools.search("return window")


# --------------------------------------------------------------------------- #
# loop: forces BOTH tools then finishes
# --------------------------------------------------------------------------- #
def test_loop_drives_both_tools_then_final_answer():
    llm = FakeListChatModel(responses=[
        "Thought: look up the window\nAction: search\nAction Input: return window",
        "Thought: double it\nAction: calculator\nAction Input: 30*2",
        "Thought: I have the answer\nFinal Answer: 60",
    ])
    result = react.run_react("return window doubled?", llm=llm, tools=build_tools(), max_steps=6)

    assert result.stopped_reason == "final_answer"
    assert result.answer == "60"
    assert [s.action for s in result.steps] == ["search", "calculator"]
    assert result.steps[0].action_input == "return window"
    assert "30 days" in result.steps[0].observation
    assert result.steps[1].observation == "60"


def test_loop_unsafe_calculator_input_rejected_in_observation():
    llm = FakeListChatModel(responses=[
        "Thought: nasty\nAction: calculator\nAction Input: __import__('os')",
        "Thought: ok\nFinal Answer: blocked",
    ])
    result = react.run_react("x", llm=llm, tools=build_tools(), max_steps=6)
    assert "unsafe" in result.steps[0].observation.lower()
    assert result.stopped_reason == "final_answer"


def test_loop_hits_max_steps_when_never_finishing():
    # FakeListChatModel cycles its responses, so this always emits an Action.
    llm = FakeListChatModel(responses=[
        "Thought: keep going\nAction: search\nAction Input: warranty"
    ])
    result = react.run_react("loop", llm=llm, tools=build_tools(), max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


def test_unknown_tool_surfaces_error_observation():
    llm = FakeListChatModel(responses=[
        "Thought: bad\nAction: teleport\nAction Input: moon",
        "Thought: done\nFinal Answer: nope",
    ])
    result = react.run_react("x", llm=llm, tools=build_tools(), max_steps=6)
    assert "unknown tool" in result.steps[0].observation.lower()


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel injected, no network)
# --------------------------------------------------------------------------- #
def make_client(responses: list[str]) -> TestClient:
    app.state.llm = FakeListChatModel(responses=responses)
    app.state.tools = build_tools()
    return TestClient(app)


def test_health():
    client = make_client(["Final Answer: unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "08-autonomous-react",
    }


def test_run_returns_answer_steps_and_reason():
    client = make_client([
        "Thought: search\nAction: search\nAction Input: return window",
        "Thought: double\nAction: calculator\nAction Input: 30*2",
        "Thought: done\nFinal Answer: 60",
    ])
    resp = client.post("/run", json={"task": "return window doubled?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "60"
    assert body["stopped_reason"] == "final_answer"
    assert [s["action"] for s in body["steps"]] == ["search", "calculator"]


# --------------------------------------------------------------------------- #
# Stop-sequence handling (the gateway's Anthropic path 500s on `stop`)
# --------------------------------------------------------------------------- #
def test_supports_stop_is_false_only_for_claude_models():
    assert llm_mod.model_profile("qwen-local-instruct")["supports_stop"] is True
    assert llm_mod.model_profile("qwen3:1.7b")["supports_stop"] is True
    assert llm_mod.model_profile("claude-haiku")["supports_stop"] is False


def test_stop_sequences_resolves_from_the_client_then_settings():
    """Regression: `stop` must be omitted for claude-*, which 500s on it."""

    class FakeClient:
        model_name = "claude-haiku"

    assert llm_mod.stop_sequences(FakeClient()) is None

    class LocalClient:
        model_name = "qwen-local-instruct"

    assert llm_mod.stop_sequences(LocalClient()) == list(llm_mod.STOP_MARKERS)


def test_truncate_at_stop_cuts_hallucinated_observation():
    text = "Thought: compute\nAction: calculator\nObservation: fabricated!"
    assert llm_mod.truncate_at_stop(text) == "Thought: compute\nAction: calculator"
    assert llm_mod.truncate_at_stop("Thought: done  ") == "Thought: done"


def test_loop_ignores_a_model_supplied_observation():
    """With `stop` unsupported the model may write its own Observation; the loop
    must cut it rather than treat the fabrication as a real tool result."""
    llm = FakeListChatModel(responses=[
        "Thought: look it up\nAction: search\nAction Input: return window\n"
        "Observation: the window is 999 days\nFinal Answer: 999",
        "Thought: done\nFinal Answer: 30",
    ])
    result = react.run_react("return window?", llm=llm, tools=build_tools(), max_steps=4)

    assert result.answer != "999"
    assert "999" not in result.steps[0].observation


# --------------------------------------------------------------------------- #
# Tracing — attached via LangChain callbacks. See docs/trace-format.md
# --------------------------------------------------------------------------- #
_TRACE_SCRIPT = [
    "Thought: look it up\nAction: search\nAction Input: return window",
    "Thought: done\nFinal Answer: 30 days",
]


def test_trace_absent_unless_requested():
    body = make_client(_TRACE_SCRIPT).post("/run", json={"task": "x"}).json()
    assert body["trace"] is None


def test_trace_records_llm_and_tool_spans_via_callbacks():
    """The loop is untouched: these spans come from LangChain's own callbacks."""
    trace = make_client(_TRACE_SCRIPT).post(
        "/run?trace=1", json={"task": "return window?"}
    ).json()["trace"]

    assert trace["schema_version"] == 1
    assert trace["approach"] == "langchain"
    assert [(s["seq"], s["type"], s["name"]) for s in trace["spans"]] == [
        (1, "llm", "chat"),
        (2, "tool", "search"),
        (3, "llm", "chat"),
    ]
    assert trace["outcome"]["stop_reason"] == "final_answer"
    assert trace["outcome"]["tool_calls"] == 1


def test_trace_uses_otel_role_names_not_langchain_ones():
    """LangChain says human/ai; the trace must say user/assistant.

    Without this, a langchain trace would not line up field-for-field with a
    raw-api one — which is the entire purpose of a shared format.
    """
    trace = make_client(_TRACE_SCRIPT).post(
        "/run?trace=1", json={"task": "x"}
    ).json()["trace"]

    roles = {m["role"] for s in trace["spans"] if s["type"] == "llm"
             for m in s["request"]["messages"]}
    assert roles <= {"system", "user", "assistant", "tool"}
    assert "human" not in roles and "ai" not in roles


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
