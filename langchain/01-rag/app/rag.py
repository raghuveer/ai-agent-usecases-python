"""RAG core: ingest the bundled corpus into Chroma, retrieve, and answer.

The retrieval chain is plain LCEL so the pieces stay visible:

    {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser

``answer_question`` runs that chain AND returns the source documents the
retriever selected, so the HTTP layer can report ``sources``.

Embeddings use chromadb's default embedding function (ONNX MiniLM) — no torch,
no sentence-transformers. ``langchain-chroma`` uses that default when no
``embedding`` is passed to the ``Chroma`` store, but to keep retrieval explicit
and ingestion/query consistent we wrap it via ``ChromaDefaultEmbeddings``.
"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import Settings, get_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "You are a support assistant for Northwind Robotics. Answer the question "
    "using ONLY the provided context. If the context does not contain the "
    "answer, say you don't know. Cite the source file names you used."
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _system_prompt(model: str) -> str:
    """Disable qwen3 thinking mode by prefixing /no_think."""
    if model.startswith("qwen3"):
        return "/no_think\n" + SYSTEM_PROMPT
    return SYSTEM_PROMPT


def load_corpus(data_dir: Path | None = None) -> list[Document]:
    """Read every ``*.md`` file under ``data/`` into Documents."""
    data_dir = data_dir or DATA_DIR
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_vectorstore(settings: Settings | None = None) -> Chroma:
    """Ingest the corpus into a persistent Chroma store and return it.

    Uses chromadb's default embedding function (no embedding arg passed).
    Re-ingest is idempotent-ish for this tiny demo: we add documents only if
    the collection is empty.
    """
    settings = settings or get_settings()
    store = Chroma(
        collection_name="northwind",
        persist_directory=settings.chroma_dir,
    )
    if store._collection.count() == 0:
        chunks = split_documents(load_corpus())
        store.add_documents(chunks)
    return store


def format_docs(docs: list[Document]) -> str:
    """Render retrieved docs into a prompt-friendly, source-labelled block."""
    return "\n\n".join(
        f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


def build_chain(retriever: BaseRetriever, llm: BaseChatModel, model_name: str):
    """Assemble the LCEL retrieval chain (question -> answer string)."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_prompt(model_name)), ("human", USER_PROMPT)]
    )
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_question(
    question: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    model_name: str,
) -> tuple[str, list[Document]]:
    """Return (answer, source_documents) for a question.

    We fetch the source docs once (for the response payload) and run the LCEL
    chain (which retrieves again internally) for the answer. For this tiny demo
    the double lookup is cheap and keeps the chain idiomatic.
    """
    sources = retriever.invoke(question)
    chain = build_chain(retriever, llm, model_name)
    answer = chain.invoke(question)
    return answer, sources
