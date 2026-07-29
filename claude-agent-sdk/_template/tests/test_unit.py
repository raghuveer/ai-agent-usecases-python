# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — claude-agent-sdk project template. See claude-agent-sdk/_template/README.md
"""Unit tests for the claude-agent-sdk template. Stubbed runner, no network.

Two layers are covered, and the split is the point:

* :func:`app.agent.collect` is tested against **real SDK message objects**, so
  the parsing logic is verified against the types the CLI actually emits.
* The FastAPI route is tested with a stub runner, so no subprocess is spawned.

Neither needs a key, a gateway, or Node.
"""
from __future__ import annotations

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from fastapi.testclient import TestClient

from app.agent import AgentResult, collect
from app.main import create_app


def make_client(result: AgentResult | None = None) -> TestClient:
    answer = result or AgentResult(text="Paris.", num_turns=1, cost_usd=0.0001)

    async def stub_runner(prompt: str, options) -> AgentResult:
        return answer

    return TestClient(create_app(runner=stub_runner))


async def _stream(*messages):
    for m in messages:
        yield m


def _assistant(*blocks) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-haiku")


def _result(**kw) -> ResultMessage:
    defaults = dict(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="s1",
        total_cost_usd=0.0002,
    )
    defaults.update(kw)
    return ResultMessage(**defaults)


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "claude-agent-sdk",
        "usecase": "_template",
    }


def test_run_returns_agent_answer():
    resp = make_client().post("/run", json={"question": "Capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Paris."
    assert body["num_turns"] == 1


@pytest.mark.anyio
async def test_collect_joins_text_and_records_tool_calls():
    stream = _stream(
        _assistant(TextBlock(text="Looking it up.")),
        _assistant(ToolUseBlock(id="t1", name="Read", input={"file_path": "a.txt"})),
        _assistant(TextBlock(text="Paris.")),
        _result(num_turns=3, total_cost_usd=0.005),
    )
    out = await collect(stream)
    assert out.text == "Looking it up.\nParis."
    assert out.tool_names == ["Read"]
    assert out.tool_calls[0].input == {"file_path": "a.txt"}
    assert out.num_turns == 3
    assert out.cost_usd == 0.005
    assert out.is_error is False


@pytest.mark.anyio
async def test_collect_falls_back_to_result_text_when_no_assistant_text():
    out = await collect(_stream(_result(result="done via tool")))
    assert out.text == "done via tool"


@pytest.mark.anyio
async def test_collect_marks_errors_and_stop_reason():
    out = await collect(
        _stream(_result(is_error=True, stop_reason="max_turns", total_cost_usd=None))
    )
    assert out.is_error is True
    assert out.stop_reason == "max_turns"
    # None cost must normalise to 0.0, not propagate None into the response model.
    assert out.cost_usd == 0.0


@pytest.mark.anyio
async def test_collect_drops_thinking_blocks():
    from claude_agent_sdk import ThinkingBlock

    out = await collect(
        _stream(
            _assistant(
                ThinkingBlock(thinking="secret reasoning", signature="sig"),
                TextBlock(text="visible"),
            ),
            _result(),
        )
    )
    assert out.text == "visible"
    assert "secret" not in out.text
