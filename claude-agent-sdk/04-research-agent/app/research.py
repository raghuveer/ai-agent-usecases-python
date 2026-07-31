# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""A research agent with two retrieval modes, one of which is air-gap safe.

The Agent SDK ships `WebSearch` and `WebFetch` as built-in tools, so the online
version of this use case needs no search-API client, no HTML extraction, and no
citation plumbing. But this repository must also run air-gapped, so web access is
**opt-in**:

* ``offline`` (default) — `Grep` / `Glob` / `Read` over the bundled corpus.
  No network. Deterministic. This is what the tests run.
* ``web`` (``RESEARCH_ALLOW_WEB=1``) — the offline tools *plus* `WebSearch` and
  `WebFetch`.

Defaulting to offline is deliberate: a live-web default would make the example
non-reproducible and would quietly break on an isolated host. The mode is
reported in the response so a reader never has to guess which one ran.

Citations are recovered from what the agent actually opened — file names from
`Read`, URLs from `WebFetch` — rather than from prose it wrote, so a claimed
source that was never fetched does not appear.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import Settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data"

OFFLINE_TOOLS = ["Grep", "Glob", "Read"]
WEB_TOOLS = ["WebSearch", "WebFetch"]

BASE_PROMPT = """You are a research assistant. Answer the question thoroughly but
concisely (under 250 words), and cite your sources.

Method: search first, read what looks relevant, then synthesise. If sources
disagree, say so. If you cannot establish something, say that rather than
guessing."""

OFFLINE_PROMPT = (
    BASE_PROMPT
    + """

You have NO internet access. Work only from the documents in the current
directory, using Grep, Glob, and Read. Cite the filenames you used. If the corpus
does not answer the question, say so plainly."""
)

WEB_PROMPT = (
    BASE_PROMPT
    + """

You may search the web (WebSearch) and fetch pages (WebFetch), and you may also
consult the local documents in the current directory with Grep/Glob/Read. Prefer
primary sources. Cite URLs for anything you took from the web and filenames for
anything local."""
)


@dataclass
class ResearchResult:
    answer: str
    mode: str
    citations: list[str]
    searches: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str = "end_turn"


def _citations_from(tool_calls) -> list[str]:
    """Sources the agent actually opened, de-duplicated in first-use order."""
    out: list[str] = []
    for call in tool_calls:
        value: str | None = None
        if call.name == "Read":
            raw = call.input.get("file_path") or call.input.get("path")
            value = Path(str(raw)).name if raw else None
        elif call.name == "WebFetch":
            raw = call.input.get("url")
            value = str(raw) if raw else None
        if value and value not in out:
            out.append(value)
    return out


def _searches_from(tool_calls) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        if call.name == "Grep" and call.input.get("pattern"):
            out.append(str(call.input["pattern"]))
        elif call.name == "WebSearch" and call.input.get("query"):
            out.append(str(call.input["query"]))
    return out


async def research(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> ResearchResult:
    runner = runner or default_runner
    web = settings.research_allow_web
    tools = OFFLINE_TOOLS + (WEB_TOOLS if web else [])

    options = build_options(
        settings,
        system_prompt=WEB_PROMPT if web else OFFLINE_PROMPT,
        allowed_tools=tools,
        tools=tools,
        cwd=str(corpus_dir or CORPUS_DIR),
        # Research is iterative: search, read, refine.
        max_turns=max(settings.agent_max_turns, 8),
    )
    result = await runner(question, options)

    return ResearchResult(
        answer=result.text,
        mode="web" if web else "offline",
        citations=_citations_from(result.tool_calls),
        searches=_searches_from(result.tool_calls),
        tools_used=result.tool_names,
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
        stop_reason=outcome_of(result),
    )
