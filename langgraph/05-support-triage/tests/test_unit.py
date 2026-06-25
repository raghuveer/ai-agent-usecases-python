"""Unit tests for UC5 support-triage (langgraph). Fake LLM, no network.

``FakeListChatModel`` returns its ``responses`` in order, one per ``invoke``.
A graph pass calls classify then exactly one specialist node, so we supply
responses in (classification, reply) pairs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import triage
from app.main import create_app


@pytest.fixture(autouse=True)
def _clear_memory():
    triage.reset_memory()
    yield
    triage.reset_memory()


def make_client(responses: list[str]) -> TestClient:
    llm = FakeListChatModel(responses=responses)
    return TestClient(create_app(llm=llm))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_classification_plain_json():
    intent, conf = triage.parse_classification('{"intent": "billing", "confidence": 0.9}')
    assert intent == "billing"
    assert conf == 0.9


def test_parse_classification_handles_fenced_and_prose():
    raw = 'ok:\n```json\n{"intent":"technical","confidence":0.66}\n```'
    intent, conf = triage.parse_classification(raw)
    assert intent == "technical"
    assert conf == 0.66


def test_parse_classification_unknown_falls_back():
    intent, conf = triage.parse_classification('{"intent":"sales","confidence":0.3}')
    assert intent == "general"
    assert conf == 0.3


def test_parse_classification_garbage_is_safe():
    intent, conf = triage.parse_classification("no json here")
    assert intent == "general"
    assert conf == 0.0


# --------------------------------------------------------------------------- #
# Graph routing + escalation
# --------------------------------------------------------------------------- #
def test_graph_routes_to_billing_specialist_high_confidence():
    llm = FakeListChatModel(
        responses=['{"intent": "billing", "confidence": 0.95}', "Billing reply."]
    )
    graph = triage.build_triage_graph(llm)
    out = triage.run_triage(graph, "I was double charged", session_id=None)
    assert out["intent"] == "billing"
    assert out["response"] == "Billing reply."
    assert out["escalate"] is False


def test_graph_routes_to_technical_specialist():
    llm = FakeListChatModel(
        responses=['{"intent": "technical", "confidence": 0.8}', "Tech reply."]
    )
    graph = triage.build_triage_graph(llm)
    out = triage.run_triage(graph, "app crashes on launch", session_id=None)
    assert out["intent"] == "technical"
    assert out["response"] == "Tech reply."
    assert out["escalate"] is False


def test_low_confidence_escalates():
    llm = FakeListChatModel(
        responses=['{"intent": "general", "confidence": 0.2}', "Hmm."]
    )
    graph = triage.build_triage_graph(llm)
    out = triage.run_triage(graph, "??", session_id=None)
    assert out["intent"] == "general"
    assert out["confidence"] == 0.2
    assert out["escalate"] is True


def test_route_intent_helper_returns_node_name():
    state = {"intent": "technical"}
    assert triage.route_intent(state) == "technical"  # type: ignore[arg-type]


def test_session_memory_feeds_history_into_next_turn():
    llm = FakeListChatModel(
        responses=[
            '{"intent": "billing", "confidence": 0.9}',
            "First reply.",
            '{"intent": "billing", "confidence": 0.9}',
            "Second reply.",
        ]
    )
    graph = triage.build_triage_graph(llm)
    triage.run_triage(graph, "I have a billing question", session_id="s1")
    history = triage.get_history("s1")
    assert any("I have a billing question" in t["content"] for t in history)
    assert any(t["content"] == "First reply." for t in history)


# --------------------------------------------------------------------------- #
# HTTP level
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "05-support-triage",
    }


def test_run_classifies_and_responds_offline():
    client = make_client(
        ['{"intent": "billing", "confidence": 0.92}', "I can help with the double charge."]
    )
    resp = client.post("/run", json={"message": "I was double charged on my invoice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "billing"
    assert body["confidence"] == 0.92
    assert body["escalate"] is False
    assert "double charge" in body["response"]
