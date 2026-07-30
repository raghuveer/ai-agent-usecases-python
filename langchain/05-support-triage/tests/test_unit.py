# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langchain). See langchain/05-support-triage/README.md
"""Unit tests for UC5 support-triage (langchain) — fully mocked, no network.

``FakeListChatModel`` returns its ``responses`` in order, one per ``invoke``.
Each triage pass invokes the classifier chain then the responder chain, so we
supply responses in (classification, reply) pairs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import triage
from app.main import app


@pytest.fixture(autouse=True)
def _clear_memory():
    triage.reset_memory()
    yield
    triage.reset_memory()


def make_client(responses: list[str]) -> TestClient:
    app.state.llm = FakeListChatModel(responses=responses)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Parsing + routing + escalation
# --------------------------------------------------------------------------- #
def test_parse_classification_plain_json():
    intent, conf = triage.parse_classification('{"intent": "billing", "confidence": 0.9}')
    assert intent == "billing"
    assert conf == 0.9


def test_parse_classification_handles_fenced_and_prose():
    raw = 'Sure:\n```json\n{"intent":"technical","confidence":0.7}\n```'
    intent, conf = triage.parse_classification(raw)
    assert intent == "technical"
    assert conf == 0.7


def test_parse_classification_unknown_falls_back():
    intent, conf = triage.parse_classification('{"intent":"sales","confidence":0.3}')
    assert intent == "general"
    assert conf == 0.3


def test_parse_classification_garbage_is_safe():
    intent, conf = triage.parse_classification("totally not json")
    assert intent == "general"
    assert conf == 0.0


def test_high_confidence_routes_billing_no_escalation():
    llm = FakeListChatModel(
        responses=['{"intent": "billing", "confidence": 0.95}', "I can help with that invoice."]
    )
    result = triage.triage(
        "I was double charged",
        session_id=None,
        llm=llm,
        escalate_threshold=0.5,
    )
    assert result["intent"] == "billing"
    assert result["response"] == "I can help with that invoice."
    assert result["escalate"] is False


def test_low_confidence_escalates():
    llm = FakeListChatModel(
        responses=['{"intent": "technical", "confidence": 0.2}', "Let me look."]
    )
    result = triage.triage(
        "weird thing happening",
        session_id=None,
        llm=llm,
        escalate_threshold=0.5,
    )
    assert result["intent"] == "technical"
    assert result["confidence"] == 0.2
    assert result["escalate"] is True


def test_responder_chain_uses_specialist_prompt():
    """The responder chain for an intent must carry that specialist system text."""
    llm = FakeListChatModel(responses=["x"])
    chain = triage.build_responder_chain(llm, "billing")
    rendered = chain.steps[0].invoke({"message": "hi", "history": ""})
    system_text = rendered.messages[0].content
    assert "billing-support specialist" in system_text


def test_session_memory_feeds_history_into_next_turn():
    # Two passes -> four responses (classify, reply) x 2.
    llm = FakeListChatModel(
        responses=[
            '{"intent": "billing", "confidence": 0.9}',
            "First reply.",
            '{"intent": "billing", "confidence": 0.9}',
            "Second reply.",
        ]
    )
    triage.triage("I have a billing question", session_id="s1", llm=llm, escalate_threshold=0.5)
    # The classifier chain for turn 2 should now see turn-1 history.
    history = triage.get_history("s1")
    assert any("I have a billing question" in t["content"] for t in history)
    chain = triage.build_classifier_chain(llm)
    rendered = chain.steps[0].invoke(
        {"message": "and the refund?", "history": triage.render_history(history)}
    )
    human_text = rendered.messages[-1].content
    assert "I have a billing question" in human_text


# --------------------------------------------------------------------------- #
# HTTP level
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "05-support-triage",
    }


def test_run_classifies_and_responds_offline():
    client = make_client(
        ['{"intent": "billing", "confidence": 0.92}', "I can help with the double charge."]
    )
    resp = client.post(
        "/run", json={"message": "I was double charged on my invoice"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "billing"
    assert body["confidence"] == 0.92
    assert body["escalate"] is False
    assert "double charge" in body["response"]


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
