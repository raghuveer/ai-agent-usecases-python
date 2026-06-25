# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""Unit tests for UC8 autonomous-react (raw-api) — fully mocked, no network.

We script a sequence of LLM replies (a scripted ``llm_call``) so the ReAct loop
runs offline and deterministically. The scripts force BOTH tools, exercise the
unsafe-calculator rejection, and cover both ``stopped_reason`` outcomes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import react, tools
from app.main import create_app
from app.settings import Settings


class ScriptedLLM:
    """Returns queued replies in order, recording the messages it was sent.

    When the queue is exhausted it repeats the last reply, so a one-element
    script that emits an Action will loop, and a garbled script stays garbled.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        if len(self._replies) > 1:
            return self._replies.pop(0)
        return self._replies[0]


# --------------------------------------------------------------------------- #
# tools: calculator safety
# --------------------------------------------------------------------------- #
def test_calculator_evaluates_arithmetic():
    assert tools.calculator("30 * 2") == "60"
    assert tools.calculator("99 + 1") == "100"
    assert tools.calculator("(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_input():
    import pytest

    for hostile in ["__import__('os')", "open('x')", "a + b", "1; 2"]:
        with pytest.raises(tools.UnsafeExpression):
            tools.safe_eval(hostile)


def test_search_finds_fact():
    out = tools.search("return window")
    assert "30 days" in out


def test_search_no_match():
    assert "No matching fact" in tools.search("zzz nonexistent qqq")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_action_and_final():
    text = "Thought: I should search\nAction: search\nAction Input: return window"
    assert react.parse_action(text) == ("search", "return window")
    assert react.parse_final_answer(text) is None

    final = "Thought: done\nFinal Answer: 60"
    assert react.parse_final_answer(final) == "60"


# --------------------------------------------------------------------------- #
# loop: forces BOTH tools then finishes
# --------------------------------------------------------------------------- #
def test_loop_drives_both_tools_then_final_answer():
    script = ScriptedLLM([
        "Thought: look up the window\nAction: search\nAction Input: return window",
        "Thought: double it\nAction: calculator\nAction Input: 30*2",
        "Thought: I have the answer\nFinal Answer: 60",
    ])

    result = react.run_react("return window doubled?", llm_call=script, max_steps=6)

    assert result.stopped_reason == "final_answer"
    assert result.answer == "60"
    assert [s.action for s in result.steps] == ["search", "calculator"]
    # Parsed inputs were threaded to the tools and observations captured.
    assert result.steps[0].action_input == "return window"
    assert "30 days" in result.steps[0].observation
    assert result.steps[1].action_input == "30*2"
    assert result.steps[1].observation == "60"


def test_loop_unsafe_calculator_input_is_rejected_in_observation():
    script = ScriptedLLM([
        "Thought: try something nasty\nAction: calculator\nAction Input: __import__('os')",
        "Thought: ok\nFinal Answer: blocked",
    ])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert "unsafe" in result.steps[0].observation.lower()
    assert result.stopped_reason == "final_answer"


def test_loop_hits_max_steps_when_never_finishing():
    # Always emits an Action, never a Final Answer.
    never_done = ScriptedLLM([
        "Thought: keep going\nAction: search\nAction Input: warranty"
    ] * 10)
    result = react.run_react("loop forever", llm_call=never_done, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


def test_loop_nudges_once_then_stops_on_garbled_output():
    script = ScriptedLLM(["I have no idea what format to use."])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert result.stopped_reason == "max_steps"
    # The nudge message was appended before the second (empty-script) call.
    assert len(script.calls) == 2


def test_unknown_tool_surfaces_error_observation():
    script = ScriptedLLM([
        "Thought: bad tool\nAction: teleport\nAction Input: moon",
        "Thought: done\nFinal Answer: nope",
    ])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert "unknown tool" in result.steps[0].observation.lower()


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def _fake_openai_client(replies: list[str]) -> MagicMock:
    client = MagicMock()
    queue = list(replies)

    def _create(**kwargs):
        text = queue.pop(0) if queue else "Final Answer: done"
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        return MagicMock(choices=[choice])

    client.chat.completions.create.side_effect = _create
    return client


def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client([
        "Thought: search\nAction: search\nAction Input: return window",
        "Thought: double\nAction: calculator\nAction Input: 30*2",
        "Thought: done\nFinal Answer: 60",
    ])

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "08-autonomous-react",
    }

    r = client.post("/run", json={"task": "return window doubled?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "60"
    assert body["stopped_reason"] == "final_answer"
    assert [s["action"] for s in body["steps"]] == ["search", "calculator"]
