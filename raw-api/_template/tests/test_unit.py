# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — raw-api project template. See raw-api/_template/README.md
"""Unit tests for the template — fully mocked, no network."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import llm
from app.main import create_app
from app.settings import Settings


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_apply_no_think_qwen3():
    assert llm.apply_no_think("qwen3:1.7b", "hi").startswith("/no_think")
    assert llm.apply_no_think("claude-haiku-4-5", "hi") == "hi"


def test_chat_uses_injected_client():
    client = _fake_openai_client("  echoed  ")
    out = llm.chat(
        client,
        model="qwen3:1.7b",
        system_prompt="sys",
        user_prompt="hello",
    )
    assert out == "echoed"
    # qwen3 -> /no_think prepended to the system message actually sent.
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["content"].startswith("/no_think")


def test_health_and_run_echo():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client("pong")

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {"status": "ok", "approach": "raw-api", "usecase": "_template"}

    r = client.post("/run", json={"question": "ping"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "pong"
    assert body["sources"] == []


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
