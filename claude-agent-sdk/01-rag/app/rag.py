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

**Retrieval-by-filesystem cuts both ways.** Giving an agent real file tools
means it can read real files — including ones you did not mean to expose. Asked
for an absolute path, a live agent read this project's own `.env`. Confinement
is enforced by :func:`make_corpus_gate`, not by `cwd`, which is a starting
directory and not a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from .agent import Runner, build_options, default_runner, outcome_of
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
    """What one RAG run produced, plus the retrieval trail that produced it."""

    answer: str
    sources: list[str]
    searches: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str = "end_turn"


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


# --------------------------------------------------------------------------- #
# Confining file access to the corpus (F15 in docs/security-review.md)
# --------------------------------------------------------------------------- #
# Every file tool here takes a path, and they do not spell it the same way.
PATH_ARGS = ("file_path", "path", "notebook_path")


def _resolve_under(root: Path, raw: str) -> Path:
    """Where ``raw`` would actually be read from, relative paths taken from ``root``."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    # resolve() collapses `..` *and* follows symlinks, so a link planted inside
    # the corpus cannot be used to read outside it.
    return candidate.resolve()


def make_corpus_gate(root: Path):
    """Refuse any tool call whose path reaches outside ``root``.

    ``cwd`` is where the agent *starts*, not where it is *allowed*, and the
    built-in file tools accept absolute paths. Asked for one, a live agent read
    it without hesitation — including this project's own ``.env``, which holds
    the gateway key. A question-answering endpoint that will read arbitrary
    server files on request is a credential-disclosure primitive, and it takes
    no cleverness to reach: "read /path/to/.env" is the whole attack.

    Unlike the equivalent gate in ``02-code-generation``, this one is a real
    boundary rather than a speed bump — **because this project grants no
    shell**. There, a refused ``Write`` was simply redone with ``Bash``; here
    the gated tools are the only route to the filesystem at all. The difference
    is not the gate. It is which capabilities the agent was given.
    """
    root = root.resolve()

    async def can_use_tool(
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        for key in PATH_ARGS:
            raw = input_data.get(key)
            if not raw:
                continue
            target = _resolve_under(root, str(raw))
            if target != root and not target.is_relative_to(root):
                # Not `interrupt`: let the agent retry inside the corpus and
                # still answer, rather than killing the run.
                return PermissionResultDeny(
                    message=(
                        f"Refused: {raw!r} is outside the document corpus. Only "
                        "files in the current directory can be read."
                    ),
                    interrupt=False,
                )
        # No path argument, or a relative one that stays inside. Tools called
        # without a path default to cwd, which is the corpus.
        return PermissionResultAllow(updated_input=input_data)

    return can_use_tool


async def answer(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> RagResult:
    """Answer `question` from the corpus, confined to it by the gate.

    Retrieval here is lexical (`Grep`/`Read`), not vector search, so a
    semantically-phrased question can miss where `langchain/01-rag`
    would hit. That is the trade this project exists to show.
    """
    runner = runner or default_runner
    root = (corpus_dir or CORPUS_DIR).resolve()
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        # DELIBERATELY EMPTY. Naming a tool here auto-approves it *before*
        # `can_use_tool` runs, so the gate below would never see a path.
        allowed_tools=[],
        tools=RAG_TOOLS,
        # Where the agent starts. NOT where it is allowed — that is the gate.
        cwd=str(root),
        permission_mode="default",
        can_use_tool=make_corpus_gate(root),
    )
    result = await runner(question, options)
    return RagResult(
        answer=result.text,
        sources=_sources_from(result.tool_calls),
        searches=_searches_from(result.tool_calls),
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
        stop_reason=outcome_of(result),
    )
