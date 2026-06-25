# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (raw-api). See raw-api/01-rag/README.md
"""Hand-written RAG: ingest -> retrieve -> grounded prompt -> chat call.

This is the raw-api point: nothing is hidden behind a framework chain. We
chunk the bundled corpus, store chunks in a Chroma collection (using chromadb's
default embedding function — ONNX MiniLM, no torch), retrieve the top-k by
similarity, build a grounded prompt by hand, and call the LLM.

The retriever and the LLM call are both pluggable so unit tests can swap in
fakes and never touch the network or build an index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Instruct the model to answer ONLY from the supplied context and cite sources.
SYSTEM_PROMPT = (
    "You are a support assistant for Northwind Robotics. "
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't have that information. "
    "Cite the source filenames you used in parentheses, e.g. (northwind-returns.md)."
)


@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class Retrieved:
    source: str
    snippet: str


class Retriever(Protocol):
    """Anything that maps a question to a ranked list of Retrieved chunks."""

    def retrieve(self, question: str, top_k: int) -> list[Retrieved]: ...


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def _split_paragraphs(text: str) -> list[str]:
    """Split markdown into paragraph-ish chunks on blank lines."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Read every ``*.md`` under ``data_dir`` and split into chunks."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        for para in _split_paragraphs(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(source=path.name, text=para))
    return chunks


# --------------------------------------------------------------------------- #
# Chroma-backed retriever
# --------------------------------------------------------------------------- #
class ChromaRetriever:
    """Retriever backed by a persistent Chroma collection.

    Uses chromadb's default embedding function (no extra deps). Re-ingests the
    bundled corpus into a fresh collection each startup so the demo is
    reproducible.
    """

    def __init__(self, chroma_dir: str, data_dir: Path = DATA_DIR):
        import chromadb

        self._client = chromadb.PersistentClient(path=chroma_dir)
        # Fresh collection each boot keeps the demo deterministic.
        try:
            self._client.delete_collection("northwind")
        except Exception:
            pass
        self._collection = self._client.create_collection("northwind")

        chunks = load_chunks(data_dir)
        self._collection.add(
            ids=[f"{c.source}:{i}" for i, c in enumerate(chunks)],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def retrieve(self, question: str, top_k: int) -> list[Retrieved]:
        res = self._collection.query(query_texts=[question], n_results=top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [
            Retrieved(source=m["source"], snippet=d) for d, m in zip(docs, metas)
        ]

    def close(self) -> None:
        """Release the underlying Chroma client (frees file handles on Windows)."""
        try:
            self._client._system.stop()  # type: ignore[attr-defined]
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Prompt building + answer
# --------------------------------------------------------------------------- #
def build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Build the grounded user prompt from retrieved context."""
    if retrieved:
        context = "\n\n".join(
            f"[{r.source}]\n{r.snippet}" for r in retrieved
        )
    else:
        context = "(no context retrieved)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite the source filenames."
    )


# An LLM call is `(system_prompt, user_prompt) -> answer`. Injectable for tests.
LLMCall = Callable[[str, str], str]


def answer(
    question: str,
    *,
    retriever: Retriever,
    llm_call: LLMCall,
    top_k: int,
) -> dict:
    """Full RAG pass: retrieve -> prompt -> LLM. Returns answer + sources."""
    retrieved = retriever.retrieve(question, top_k)
    user_prompt = build_prompt(question, retrieved)
    text = llm_call(SYSTEM_PROMPT, user_prompt)
    return {
        "answer": text,
        "sources": [{"source": r.source, "snippet": r.snippet} for r in retrieved],
    }
