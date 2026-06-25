"""RAG core for UC1 as a tiny langgraph StateGraph.

Pieces:
- ``ingest_corpus`` builds (or loads) a Chroma vector store from the bundled
  ``data/*.md`` files using chromadb's default embedding function (ONNX MiniLM —
  no torch / sentence-transformers).
- ``build_rag_graph`` compiles a ``StateGraph`` with ``retrieve`` -> ``generate``
  -> END. Both the retriever and the LLM are injected, so unit tests can swap in
  fakes and run fully offline.

The graph is intentionally tiny: RAG is overkill on a graph. It is modelled this
way only so the langgraph approach can be compared apples-to-apples with the
raw-api and langchain versions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import system_prompt
from .settings import Settings, get_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLECTION = "northwind"

SYSTEM_INSTRUCTION = (
    "You are a support assistant for Northwind Robotics. Answer the question "
    "using ONLY the provided context. If the answer is not in the context, say "
    "you don't know. Cite the source filename(s) you used in parentheses."
)


# --- Retriever abstraction (injectable) ------------------------------------

class Retriever(Protocol):
    """Anything that turns a query into a list of Documents."""

    def invoke(self, query: str) -> list[Document]:  # pragma: no cover - protocol
        ...


def load_documents() -> list[Document]:
    """Load the bundled markdown corpus into one Document per file."""
    docs: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append(
            Document(page_content=path.read_text(encoding="utf-8"),
                     metadata={"source": path.name})
        )
    return docs


class DefaultEmbeddings:
    """Adapt chromadb's default embedding function to the langchain Embeddings API.

    chromadb's default function is ONNX MiniLM (all-MiniLM-L6-v2) and needs no
    torch / sentence-transformers. langchain-chroma expects a langchain
    ``Embeddings`` object, so we wrap it with the two methods it calls.
    """

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._fn([text])[0]]


def ingest_corpus(settings: Settings | None = None, top_k: int | None = None):
    """Build/load a Chroma vector store and return a retriever.

    Uses chromadb's default embedding function (ONNX MiniLM, no torch). Imported
    lazily so unit tests that inject a fake retriever never need chromadb
    installed at import time.
    """
    from langchain_chroma import Chroma

    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.rag_top_k

    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=settings.chroma_dir,
        embedding_function=DefaultEmbeddings(),
    )
    if store._collection.count() == 0:
        docs = load_documents()
        store.add_texts(
            texts=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
        )
    return store.as_retriever(search_kwargs={"k": k})


# --- Graph -----------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    top_k: int
    documents: list[Document]
    answer: str


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{src}]\n{d.page_content}")
    return "\n\n".join(blocks)


def build_rag_graph(retriever: Retriever, llm: BaseChatModel,
                    settings: Settings | None = None):
    """Compile a retrieve -> generate -> END graph. Deps are injected."""
    settings = settings or get_settings()

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs}

    def generate(state: RAGState) -> dict:
        docs = state["documents"]
        context = _format_context(docs)
        messages = [
            SystemMessage(content=system_prompt(SYSTEM_INSTRUCTION, settings)),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {state['question']}"
            ),
        ]
        result = llm.invoke(messages)
        return {"answer": result.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def sources_from_docs(docs: list[Document], snippet_len: int = 200) -> list[dict]:
    """Shape retrieved docs into the {source, snippet} response items."""
    out = []
    for d in docs:
        out.append({
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content.strip().replace("\n", " ")[:snippet_len],
        })
    return out
