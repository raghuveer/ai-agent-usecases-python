# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""Unit tests for UC04 research-agent (claude-agent-sdk). Stubbed agent, no network.

The important assertion is the air-gap guarantee: in the default offline mode the
web tools must not be on the allow-list at all, so an agent physically cannot
reach the network.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.research import CORPUS_DIR, research
from app.settings import Settings

ANSWER = "Local Qwen models could not sustain a text ReAct loop."


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=ANSWER,
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name="Grep", input={"pattern": "ReAct"}),
                ToolCall(name="Read", input={"file_path": "/c/model-findings.md"}),
            ],
            num_turns=4,
            cost_usd=0.003,
        )

    return runner


def make_client(**kw) -> TestClient:
    return TestClient(create_app(runner=make_runner(**kw)))


def test_health_reports_mode():
    body = make_client().get("/health").json()
    assert body["status"] == "ok"
    assert body["approach"] == "claude-agent-sdk"
    assert body["usecase"] == "04-research-agent"
    assert body["mode"] == "offline"


def test_run_returns_answer_with_citations_and_searches():
    body = make_client().post("/run", json={"question": "Can Qwen do ReAct?"}).json()
    assert body["answer"] == ANSWER
    assert body["mode"] == "offline"
    assert body["citations"] == ["model-findings.md"]
    assert body["searches"] == ["ReAct"]


def test_citations_include_urls_when_web_tools_were_used():
    calls = [
        ToolCall(name="WebSearch", input={"query": "react agents"}),
        ToolCall(name="WebFetch", input={"url": "https://example.com/a"}),
        ToolCall(name="Read", input={"file_path": "/c/local.md"}),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == ["https://example.com/a", "local.md"]
    assert body["searches"] == ["react agents"]


def test_citations_are_deduped_in_first_use_order():
    calls = [
        ToolCall(name="Read", input={"file_path": "/c/b.md"}),
        ToolCall(name="Read", input={"file_path": "/c/a.md"}),
        ToolCall(name="Read", input={"file_path": "/c/b.md"}),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == ["b.md", "a.md"]


def test_citations_come_from_tool_calls_not_prose():
    """A source the agent never opened must not be reported as a citation."""
    calls = [ToolCall(name="Grep", input={"pattern": "x"})]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == []


@pytest.mark.anyio
async def test_offline_mode_cannot_reach_the_network():
    """The air-gap guarantee: web tools are absent from the allow-list."""
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        seen["prompt"] = options.system_prompt
        seen["cwd"] = options.cwd
        return AgentResult(text="ok")

    await research("q", Settings(research_allow_web=False), spy)

    assert set(seen["tools"]) == {"Grep", "Glob", "Read"}
    assert "WebSearch" not in seen["tools"] and "WebFetch" not in seen["tools"]
    assert "NO internet access" in seen["prompt"]
    assert Path(seen["cwd"]) == CORPUS_DIR


@pytest.mark.anyio
async def test_web_mode_adds_web_tools_and_keeps_local_ones():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        seen["prompt"] = options.system_prompt
        return AgentResult(text="ok")

    await research("q", Settings(research_allow_web=True), spy)

    assert set(seen["tools"]) == {"Grep", "Glob", "Read", "WebSearch", "WebFetch"}
    assert "may search the web" in seen["prompt"]


@pytest.mark.anyio
async def test_research_raises_turn_ceiling_for_iteration():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["max_turns"] = options.max_turns
        return AgentResult(text="ok")

    await research("q", Settings(), spy)
    assert seen["max_turns"] >= 8


def test_bundled_corpus_exists():
    assert list(CORPUS_DIR.glob("*.md")), f"no corpus documents in {CORPUS_DIR}"


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
