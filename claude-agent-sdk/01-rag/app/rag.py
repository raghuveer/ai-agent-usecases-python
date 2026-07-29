# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC01 Q&A / RAG (claude-agent-sdk). See claude-agent-sdk/01-rag/README.md
"""RAG with no vector store — the agent retrieves by searching the filesystem.

The other three approaches embed the corpus into Chroma with
`sentence-transformers` and retrieve the top-k nearest chunks. This one has no
embedding model, no index, and no chunking: the agent is given `Grep`, `Glob`,
and `Read` over a corpus directory and works out its own search strategy —
grep a term, read what looks relevant, refine, repeat.

That is a genuinely different retrieval architecture, with a real trade-off:

* **No index to build, stale, or re-embed.** Drop a file into `data/` and it is
  immediately searchable. Retrieval is lexical and exact.
* **No semantic matching.** A question phrased with different vocabulary than the
  documents may retrieve nothing, where embeddings would still match. Multi-turn
  searching partly compensates — the agent can try synonyms — but at the cost of
  more model turns per question.

Use this when the corpus is small, textual, and on disk. Use `raw-api/01-rag` or
`langchain/01-rag` when you need semantic recall over a large corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import Runner, build_options, default_runner
from .settings import Settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = """You answer questions strictly from the documents in the current directory.

Method:
1. Use Grep to find relevant terms; use Glob to see what files exist.
2. Use Read to pull the passages that matter.
3. If a search returns nothing, try different wording before giving up.

Rules:
- Answer ONLY from what the documents say. Never use outside knowledge.
- If the corpus does not contain the answer, say so plainly.
- Cite the filenames you used.
- Keep the answer under 150 words."""

RAG_TOOLS = ["Grep", "Glob", "Read"]


@dataclass
class RagResult:
    answer: str
    sources: list[str]
    searches: list[str]
    num_turns: int
    cost_usd: float


def _sources_from(tool_calls) -> list[str]:
    """Files the agent actually opened, de-duplicated, in first-read order."""
    seen: list[str] = []
    for call in tool_calls:
        if call.name != "Read":
            continue
        raw = call.input.get("file_path") or call.input.get("path")
        if not raw:
            continue
        name = Path(str(raw)).name
        if name not in seen:
            seen.append(name)
    return seen


def _searches_from(tool_calls) -> list[str]:
    """Grep patterns tried — the agent's retrieval strategy, made visible."""
    return [
        str(c.input.get("pattern", ""))
        for c in tool_calls
        if c.name == "Grep" and c.input.get("pattern")
    ]


async def answer(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> RagResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=RAG_TOOLS,
        tools=RAG_TOOLS,
        # Scopes the agent to the corpus: it can only search what lives here.
        cwd=str(corpus_dir or CORPUS_DIR),
    )
    result = await runner(question, options)
    return RagResult(
        answer=result.text,
        sources=_sources_from(result.tool_calls),
        searches=_searches_from(result.tool_calls),
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
    )
