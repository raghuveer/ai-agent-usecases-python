# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC01 Q&A / RAG (claude-agent-sdk). See claude-agent-sdk/01-rag/README.md
"""Unit tests for UC01 rag (claude-agent-sdk). Stubbed agent, no network.

Covers what this project owns: scoping the agent to the corpus, and recovering
the retrieval trail (which files were read, which patterns were searched) from
the tool calls.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.rag import CORPUS_DIR, answer
from app.settings import Settings

ANSWER = "Local Qwen models could not drive a text ReAct loop (model-findings.md)."


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=ANSWER,
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name="Grep", input={"pattern": "ReAct"}),
                ToolCall(name="Read", input={"file_path": "/corpus/model-findings.md"}),
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
        "usecase": "01-rag",
    }


def test_run_returns_answer_sources_and_searches():
    body = make_client().post("/run", json={"question": "Can Qwen do ReAct?"}).json()
    assert body["answer"] == ANSWER
    assert body["sources"] == ["model-findings.md"]
    assert body["searches"] == ["ReAct"]
    assert body["num_turns"] == 3


def test_sources_are_basenames_deduped_in_read_order():
    calls = [
        ToolCall(name="Read", input={"file_path": "/corpus/b.md"}),
        ToolCall(name="Read", input={"file_path": "/corpus/a.md"}),
        ToolCall(name="Read", input={"file_path": "/corpus/b.md"}),  # repeat
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["sources"] == ["b.md", "a.md"]


def test_sources_empty_when_agent_read_nothing():
    """A grounded answer with no reads is suspicious — surface it, don't hide it."""
    calls = [ToolCall(name="Grep", input={"pattern": "nothing"})]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["sources"] == []
    assert body["searches"] == ["nothing"]


def test_read_tool_alternate_path_key_is_handled():
    calls = [ToolCall(name="Read", input={"path": "/corpus/c.md"})]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["sources"] == ["c.md"]


@pytest.mark.anyio
async def test_agent_is_scoped_to_the_corpus_and_read_only():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["cwd"] = options.cwd
        seen["tools"] = options.allowed_tools
        return AgentResult(text="ok")

    await answer("q", Settings(), spy)

    assert Path(seen["cwd"]) == CORPUS_DIR
    assert set(seen["tools"]) == {"Grep", "Glob", "Read"}
    assert "Write" not in seen["tools"] and "Bash" not in seen["tools"]


def test_bundled_corpus_exists():
    """The example must be runnable straight from a clone."""
    docs = list(CORPUS_DIR.glob("*.md"))
    assert docs, f"no corpus documents found in {CORPUS_DIR}"


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
