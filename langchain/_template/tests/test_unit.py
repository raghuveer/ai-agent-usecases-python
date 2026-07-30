# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langchain project template. See langchain/_template/README.md
"""Unit tests for the template — fully mocked, no network.

The LLM is replaced with ``FakeListChatModel`` and injected on ``app.state``
before the app's lifespan runs.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.main import app, _system_prompt


def make_client(responses: list[str]) -> TestClient:
    fake = FakeListChatModel(responses=responses)
    app.state.llm = fake
    return TestClient(app)


def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "_template",
    }


def test_run_echoes_via_mock_llm():
    client = make_client(["hello from fake model"])
    resp = client.post("/run", json={"question": "hi", "top_k": None})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "hello from fake model"
    assert body["sources"] == []


def test_no_think_prefix_for_qwen3():
    assert _system_prompt("qwen3:1.7b").startswith("/no_think")
    assert not _system_prompt("claude-haiku-4-5").startswith("/no_think")


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
