"""Unit tests for UC1 RAG (langchain) — fully mocked, no network.

We inject a fake in-memory retriever and ``FakeListChatModel`` onto
``app.state``, plus exercise the LCEL chain directly with the same fakes.
"""
from __future__ import annotations

from typing import List

from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from app.main import app
from app.rag import _system_prompt, answer_question, build_chain, format_docs


class FakeRetriever(BaseRetriever):
    """Returns a fixed set of docs regardless of the query (no network)."""

    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.docs


FIXTURE_DOCS = [
    Document(
        page_content="Customers may return any product within 30 days of delivery.",
        metadata={"source": "northwind-returns.md"},
    ),
    Document(
        page_content="A $25 restocking fee applies to non-defective returns.",
        metadata={"source": "northwind-returns.md"},
    ),
]


def make_client(responses: list[str]) -> TestClient:
    app.state.retriever = FakeRetriever(docs=FIXTURE_DOCS)
    app.state.llm = FakeListChatModel(responses=responses)
    return TestClient(app)


def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "01-rag",
    }


def test_run_returns_answer_and_sources():
    client = make_client(["You can return within 30 days (northwind-returns.md)."])
    resp = client.post("/run", json={"question": "What is the return window?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "30 days" in body["answer"]
    assert len(body["sources"]) == 2
    assert body["sources"][0]["source"] == "northwind-returns.md"
    assert body["sources"][0]["snippet"]  # non-empty


def test_format_docs_labels_sources():
    block = format_docs(FIXTURE_DOCS)
    assert "[source: northwind-returns.md]" in block
    assert "30 days" in block


def test_chain_uses_retrieved_context():
    """The LCEL chain feeds retrieved context into the prompt and returns
    the (faked) model output — proving wiring without any network."""
    retriever = FakeRetriever(docs=FIXTURE_DOCS)
    llm = FakeListChatModel(responses=["grounded answer"])
    chain = build_chain(retriever, llm, "qwen3:1.7b")
    out = chain.invoke("anything")
    assert out == "grounded answer"


def test_answer_question_returns_sources():
    retriever = FakeRetriever(docs=FIXTURE_DOCS)
    llm = FakeListChatModel(responses=["ok"])
    answer, sources = answer_question("q", retriever, llm, "qwen3:1.7b")
    assert answer == "ok"
    assert [d.metadata["source"] for d in sources] == [
        "northwind-returns.md",
        "northwind-returns.md",
    ]


def test_no_think_prefix_for_qwen3():
    assert _system_prompt("qwen3:1.7b").startswith("/no_think")
    assert not _system_prompt("claude-haiku-4-5").startswith("/no_think")
