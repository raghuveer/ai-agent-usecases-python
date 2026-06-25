# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (langgraph). See langgraph/01-rag/README.md
"""Unit tests for UC1 RAG (langgraph). Fake retriever + fake LLM, no network."""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.main import create_app
from app.rag import build_rag_graph


class FakeRetriever:
    """In-memory retriever — returns canned docs, never touches Chroma/network."""

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def invoke(self, query: str) -> list[Document]:
        return self._docs


CANNED_DOCS = [
    Document(
        page_content="Customers may return any product within 30 days of delivery.",
        metadata={"source": "northwind-returns.md"},
    ),
    Document(
        page_content="Every Northwind robot includes a 1-year limited warranty.",
        metadata={"source": "northwind-warranty.md"},
    ),
]


def make_client(answer: str = "You may return within 30 days (northwind-returns.md).") -> TestClient:
    retriever = FakeRetriever(CANNED_DOCS)
    llm = FakeListChatModel(responses=[answer])
    return TestClient(create_app(retriever=retriever, llm=llm))


def test_health():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "01-rag",
    }


def test_run_returns_answer_and_sources():
    client = make_client()
    resp = client.post("/run", json={"question": "What is the return window?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "30 days" in body["answer"]
    sources = {s["source"] for s in body["sources"]}
    assert sources == {"northwind-returns.md", "northwind-warranty.md"}
    # Snippets are non-empty and derived from the retrieved docs.
    assert all(s["snippet"] for s in body["sources"])


def test_run_respects_top_k_passthrough():
    client = make_client()
    resp = client.post("/run", json={"question": "warranty?", "top_k": 1})
    assert resp.status_code == 200
    # Fake retriever ignores k, but the request must still round-trip cleanly.
    assert resp.json()["answer"]


def test_graph_node_order():
    """The compiled graph runs retrieve then generate, feeding docs to the LLM."""
    captured = {}

    class RecordingRetriever:
        def invoke(self, query: str) -> list[Document]:
            captured["queried"] = query
            return CANNED_DOCS

    llm = FakeListChatModel(responses=["grounded answer"])
    graph = build_rag_graph(RecordingRetriever(), llm)
    out = graph.invoke({"question": "q", "top_k": 3, "documents": [], "answer": ""})

    assert captured["queried"] == "q"
    assert out["answer"] == "grounded answer"
    assert [d.metadata["source"] for d in out["documents"]] == [
        "northwind-returns.md",
        "northwind-warranty.md",
    ]
