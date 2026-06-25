# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""Deterministic, offline corpus search — the researcher sub-agent's tool.

No network, no LLM, no embeddings: a keyword-overlap scan over the bundled
``data/corpus/*.md`` files. The researcher node calls ``research(topic)`` to
gather bullet facts that the writer then drafts from.

A corpus file is split into bullet "facts" (the ``- ...`` lines plus any other
non-heading lines). Each fact is scored by word overlap with the topic; the best
``top_k`` across all files are returned. This is intentionally simple and fully
reproducible so unit tests assert exact behaviour offline.
"""
from __future__ import annotations

import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"

# Words too common to carry signal when matching a topic to a fact.
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "for", "in", "on",
    "at", "by", "as", "it", "its", "than", "so", "because", "that", "with",
    "be", "this", "each", "into", "tell", "me", "about", "write", "summary",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def load_facts(corpus_dir: Path = CORPUS_DIR) -> list[tuple[str, str]]:
    """Return ``(source_filename, fact_line)`` for every fact in the corpus."""
    facts: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.lstrip("-").strip()
            if line:
                facts.append((path.name, line))
    return facts


def _score(topic_tokens: set[str], fact: str) -> int:
    return len(topic_tokens & _tokens(fact))


def research(topic: str, top_k: int = 4, corpus_dir: Path = CORPUS_DIR) -> list[str]:
    """Return up to ``top_k`` best-matching bullet facts for ``topic``.

    Deterministic: ties are broken by the original corpus order (stable sort).
    Returns ``[]`` when nothing in the corpus overlaps the topic.
    """
    topic_tokens = _tokens(topic)
    if not topic_tokens:
        return []
    facts = load_facts(corpus_dir)
    scored = [(s, f) for (_, f) in facts if (s := _score(topic_tokens, f)) > 0]
    # Stable: Python's sort keeps original order for equal scores.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [f for _, f in scored[:top_k]]


def format_research(facts: list[str]) -> str:
    """Render gathered facts as a bullet block for the writer / response."""
    if not facts:
        return "No relevant facts found in the corpus."
    return "\n".join(f"- {f}" for f in facts)
