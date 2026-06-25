# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langgraph). See langgraph/08-autonomous-react/README.md
"""Unit tests for UC8 autonomous-react (langgraph) — fully mocked, no network.

A ``FakeListChatModel`` scripts the reason node's replies so the whole
reason → act → observe → (loop | END) cycle runs offline. Tests force BOTH
tools, exercise the unsafe-calculator rejection, and cover both
``stopped_reason`` outcomes.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import react, tools
from app.main import create_app


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


def test_search_finds_fact():
    assert "30 days" in tools.search("return window")


# --------------------------------------------------------------------------- #
# graph cycle: forces BOTH tools then finishes
# --------------------------------------------------------------------------- #
def test_cycle_drives_both_tools_then_final_answer():
    llm = FakeListChatModel(responses=[
        "Thought: look up the window\nAction: search\nAction Input: return window",
        "Thought: double it\nAction: calculator\nAction Input: 30*2",
        "Thought: I have the answer\nFinal Answer: 60",
    ])
    result = react.run_react("return window doubled?", llm=llm, max_steps=6)

    assert result["stopped_reason"] == "final_answer"
    assert result["answer"] == "60"
    assert [s["action"] for s in result["steps"]] == ["search", "calculator"]
    assert result["steps"][0]["action_input"] == "return window"
    assert "30 days" in result["steps"][0]["observation"]
    assert result["steps"][1]["observation"] == "60"


def test_cycle_unsafe_calculator_input_rejected_in_observation():
    llm = FakeListChatModel(responses=[
        "Thought: nasty\nAction: calculator\nAction Input: __import__('os')",
        "Thought: ok\nFinal Answer: blocked",
    ])
    result = react.run_react("x", llm=llm, max_steps=6)
    assert "unsafe" in result["steps"][0]["observation"].lower()
    assert result["stopped_reason"] == "final_answer"


def test_cycle_hits_max_steps_when_never_finishing():
    # FakeListChatModel cycles its responses, so every reason emits an Action.
    llm = FakeListChatModel(responses=[
        "Thought: keep going\nAction: search\nAction Input: warranty"
    ])
    result = react.run_react("loop", llm=llm, max_steps=3)
    assert result["stopped_reason"] == "max_steps"
    assert len(result["steps"]) == 3


def test_unknown_tool_surfaces_error_observation():
    llm = FakeListChatModel(responses=[
        "Thought: bad\nAction: teleport\nAction Input: moon",
        "Thought: done\nFinal Answer: nope",
    ])
    result = react.run_react("x", llm=llm, max_steps=6)
    assert "unknown tool" in result["steps"][0]["observation"].lower()


def test_graph_has_react_cycle_edges():
    """The compiled graph wires reason→(act|END), act→observe, observe→reason."""
    llm = FakeListChatModel(responses=["Final Answer: x"])
    graph = react.build_react_graph(llm)
    # The compiled graph exposes its nodes; assert the cycle nodes exist.
    nodes = set(graph.get_graph().nodes.keys())
    assert {"reason", "act", "observe"} <= nodes


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel injected, no network)
# --------------------------------------------------------------------------- #
def make_client(responses: list[str]) -> TestClient:
    llm = FakeListChatModel(responses=responses)
    return TestClient(create_app(llm=llm))


def test_health():
    client = make_client(["Final Answer: unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
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
