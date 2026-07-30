# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langgraph project template. See langgraph/_template/README.md
"""Unit tests for the langgraph template. Mocked LLM, no network."""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.main import create_app


def make_client() -> TestClient:
    fake_llm = FakeListChatModel(responses=["echo: hello"])
    return TestClient(create_app(llm=fake_llm))


def test_health():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "_template",
    }


def test_run_echoes_via_mocked_llm():
    client = make_client()
    resp = client.post("/run", json={"question": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: hello"
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
