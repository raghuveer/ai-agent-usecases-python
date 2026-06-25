# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (raw-api). See raw-api/05-support-triage/README.md
"""Unit tests for UC5 support-triage (raw-api) — fully mocked, no network."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import triage
from app.main import create_app
from app.settings import Settings


@pytest.fixture(autouse=True)
def _clear_memory():
    """Each test starts with empty session memory."""
    triage.reset_memory()
    yield
    triage.reset_memory()


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# --------------------------------------------------------------------------- #
# Pure-function level: parsing + routing + escalation
# --------------------------------------------------------------------------- #
def test_parse_classification_plain_json():
    intent, conf = triage.parse_classification('{"intent": "billing", "confidence": 0.9}')
    assert intent == "billing"
    assert conf == 0.9


def test_parse_classification_handles_fenced_and_prose():
    raw = 'Here you go:\n```json\n{"intent":"technical","confidence":0.71}\n```'
    intent, conf = triage.parse_classification(raw)
    assert intent == "technical"
    assert conf == 0.71


def test_parse_classification_unknown_intent_falls_back_low():
    intent, conf = triage.parse_classification('{"intent": "sales", "confidence": 0.4}')
    assert intent == "general"
    assert conf == 0.4


def test_parse_classification_garbage_is_safe():
    intent, conf = triage.parse_classification("not json at all")
    assert intent == "general"
    assert conf == 0.0


def test_routing_picks_specialist_per_intent():
    """The responder must be called with the specialist system prompt for the
    classified intent — proving routing without any network."""
    captured: dict[str, str] = {}

    def classify_call(system: str, user: str) -> str:
        return '{"intent": "billing", "confidence": 0.95}'

    def respond_call(system: str, user: str) -> str:
        captured["system"] = system
        return "I can help with your invoice."

    result = triage.triage(
        "I was double charged",
        session_id=None,
        classify_call=classify_call,
        respond_call=respond_call,
        escalate_threshold=0.5,
    )
    assert result["intent"] == "billing"
    assert captured["system"] == triage.SPECIALIST_SYSTEM["billing"]
    assert result["escalate"] is False


def test_low_confidence_escalates():
    def classify_call(system: str, user: str) -> str:
        return '{"intent": "technical", "confidence": 0.2}'

    def respond_call(system: str, user: str) -> str:
        return "Let me take a look."

    result = triage.triage(
        "something is weird",
        session_id=None,
        classify_call=classify_call,
        respond_call=respond_call,
        escalate_threshold=0.5,
    )
    assert result["intent"] == "technical"
    assert result["confidence"] == 0.2
    assert result["escalate"] is True


def test_high_confidence_does_not_escalate():
    result = triage.triage(
        "hi there",
        session_id=None,
        classify_call=lambda s, u: '{"intent": "general", "confidence": 0.88}',
        respond_call=lambda s, u: "Hello! How can I help?",
        escalate_threshold=0.5,
    )
    assert result["escalate"] is False


def test_session_memory_feeds_history_into_next_turn():
    """A second turn in the same session must include the first turn's text in
    the prompts sent to the LLM."""
    seen_prompts: list[str] = []

    def classify_call(system: str, user: str) -> str:
        seen_prompts.append(user)
        return '{"intent": "billing", "confidence": 0.9}'

    def respond_call(system: str, user: str) -> str:
        return "Reply."

    triage.triage(
        "I have a billing question",
        session_id="s1",
        classify_call=classify_call,
        respond_call=respond_call,
        escalate_threshold=0.5,
    )
    triage.triage(
        "what about the refund?",
        session_id="s1",
        classify_call=classify_call,
        respond_call=respond_call,
        escalate_threshold=0.5,
    )
    # Second classification prompt should carry the first user message as history.
    assert "I have a billing question" in seen_prompts[1]


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    # First call (classify) returns JSON; second call (respond) returns prose.
    client_stub = MagicMock()

    def _create(**kwargs):
        # Distinguish classifier vs responder by the system prompt content.
        system = kwargs["messages"][0]["content"]
        text = (
            '{"intent": "billing", "confidence": 0.92}'
            if "classification engine" in system
            else "I can help you with the double charge."
        )
        msg = MagicMock()
        msg.content = text
        return MagicMock(choices=[MagicMock(message=msg)])

    client_stub.chat.completions.create.side_effect = _create
    app.state.client = client_stub

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "05-support-triage",
    }

    r = client.post("/run", json={"message": "I was double charged on my invoice"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "billing"
    assert body["confidence"] == 0.92
    assert body["escalate"] is False
    assert "double charge" in body["response"]
