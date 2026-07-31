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
from typing import Any

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

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
    """The report, its citations, and which mode actually ran."""

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

    A path gate is not an exfiltration boundary on its own. With
    ``RESEARCH_ALLOW_WEB=1`` this agent also holds ``WebFetch``, and a URL can
    carry data in its query string — so corpus content could leave that way even
    with every read confined. Web mode is off by default for exactly this class
    of reason; treat it as a deliberate widening of trust.
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


async def research(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> ResearchResult:
    """Research `question` over the corpus, and the web if enabled.

    Citations are recovered from what the agent actually opened, not
    from prose it wrote, so a claimed source it never fetched does not
    appear. Web mode widens trust — see `make_corpus_gate`.
    """
    runner = runner or default_runner
    web = settings.research_allow_web
    tools = OFFLINE_TOOLS + (WEB_TOOLS if web else [])

    root = (corpus_dir or CORPUS_DIR).resolve()
    options = build_options(
        settings,
        system_prompt=WEB_PROMPT if web else OFFLINE_PROMPT,
        # DELIBERATELY EMPTY — an entry here auto-approves before the gate runs.
        allowed_tools=[],
        tools=tools,
        # Where the agent starts. NOT where it is allowed — that is the gate.
        cwd=str(root),
        permission_mode="default",
        can_use_tool=make_corpus_gate(root),
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
