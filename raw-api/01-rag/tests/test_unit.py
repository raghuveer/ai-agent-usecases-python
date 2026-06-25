# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (raw-api). See raw-api/01-rag/README.md
"""Unit tests for UC1 RAG (raw-api) — fully mocked, no network, no index build."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import llm, rag
from app.main import create_app
from app.settings import Settings


class FakeRetriever:
    """Returns canned chunks, regardless of the question."""

    def __init__(self, chunks: list[rag.Retrieved]):
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[rag.Retrieved]:
        self.calls.append((question, top_k))
        return self._chunks[:top_k]


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# --------------------------------------------------------------------------- #
# Pure-function level
# --------------------------------------------------------------------------- #
def test_load_chunks_reads_bundled_corpus():
    chunks = rag.load_chunks()
    sources = {c.source for c in chunks}
    assert {
        "northwind-products.md",
        "northwind-returns.md",
        "northwind-warranty.md",
        "northwind-support.md",
    } <= sources
    assert all(c.text for c in chunks)


def test_build_prompt_includes_context_and_sources():
    retrieved = [
        rag.Retrieved(source="northwind-returns.md", snippet="30 day return window"),
    ]
    prompt = rag.build_prompt("can I return it?", retrieved)
    assert "northwind-returns.md" in prompt
    assert "30 day return window" in prompt
    assert "can I return it?" in prompt


def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("claude-haiku-4-5", "x") == "x"


def test_answer_uses_retriever_and_llm_call():
    retriever = FakeRetriever(
        [
            rag.Retrieved(source="northwind-returns.md", snippet="full refund in 30 days"),
            rag.Retrieved(source="northwind-warranty.md", snippet="1-year warranty"),
        ]
    )
    captured = {}

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "You get a full refund within 30 days (northwind-returns.md)."

    result = rag.answer(
        "what is the return policy?",
        retriever=retriever,
        llm_call=fake_llm,
        top_k=2,
    )

    assert retriever.calls == [("what is the return policy?", 2)]
    assert "full refund" in result["answer"]
    assert [s["source"] for s in result["sources"]] == [
        "northwind-returns.md",
        "northwind-warranty.md",
    ]
    # Retrieved context made it into the prompt sent to the model.
    assert "full refund in 30 days" in captured["user"]
    assert "ONLY the provided context" in captured["system"]


# --------------------------------------------------------------------------- #
# HTTP level (mocked retriever + mocked openai client, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(
        "Returns get a full refund within 30 days (northwind-returns.md)."
    )
    # Pre-set the retriever so the startup hook does NOT build a Chroma index.
    app.state.retriever = FakeRetriever(
        [rag.Retrieved(source="northwind-returns.md", snippet="full refund in 30 days")]
    )

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {"status": "ok", "approach": "raw-api", "usecase": "01-rag"}

    r = client.post("/run", json={"question": "return policy?", "top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert "full refund" in body["answer"]
    assert body["sources"] == [
        {"source": "northwind-returns.md", "snippet": "full refund in 30 days"}
    ]
    # The grounded system prompt reached the model. (qwen3 /no_think prefixing is
    # covered independently by test_apply_no_think_qwen3_only; here the model comes
    # from .env so it isn't necessarily qwen3.)
    sent = app.state.client.chat.completions.create.call_args.kwargs["messages"]
    assert "ONLY the provided context" in sent[0]["content"]
