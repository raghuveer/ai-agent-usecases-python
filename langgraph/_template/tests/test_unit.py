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
