# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""Multi-agent orchestration via the SDK's native subagents.

`ClaudeAgentOptions.agents` takes a dict of :class:`AgentDefinition`s. The lead
agent then delegates to them with the built-in `Agent` tool, and the SDK handles
spawning, isolating, and collecting each one:

    lead ──► Agent(researcher) ──► findings ┐
         ──► Agent(analyst)   ──► analysis  ├──► lead writes the report
         ──► Agent(writer)    ──► prose     ┘

Each subagent gets its **own context window and its own tool allow-list**. That
last part is the real payoff and is hard to reproduce elsewhere: `researcher` can
read the corpus but cannot write; `writer` has no file tools at all and must work
from what it is handed. Least privilege per role, declared in data.

Contrast the siblings: `langgraph/07` wires sub-graphs and shared state by hand,
`langchain/07` is a workaround, and `raw-api/07` hand-rolls the orchestrator and
is explicitly marked impractical. Here the orchestration is a dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import Settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data"

LEAD_PROMPT = """You lead a small research team working from a local document corpus.

For the user's question:
1. Delegate corpus gathering to the `researcher` subagent.
2. Delegate interpretation of those findings to the `analyst` subagent.
3. Delegate the final write-up to the `writer` subagent.
4. Return the writer's report as your final answer.

Use the Agent tool to delegate. Do not read the corpus yourself — that is the
researcher's job. Keep the final report under 200 words."""

TEAM: dict[str, AgentDefinition] = {
    "researcher": AgentDefinition(
        description="Searches the local document corpus and returns raw findings with file citations.",
        prompt=(
            "You search a local corpus of markdown documents. Use Grep and Glob to "
            "locate relevant passages and Read to pull them. Return concise bullet "
            "findings, each with the source filename. Do not interpret or "
            "speculate — report only what the documents say."
        ),
        # Read-only: this role must never mutate the corpus.
        tools=["Grep", "Glob", "Read"],
    ),
    "analyst": AgentDefinition(
        description="Interprets findings, identifies themes, tensions, and gaps.",
        prompt=(
            "You interpret research findings. Identify themes, contradictions, and "
            "gaps. Be explicit about what the findings do NOT establish. Do not "
            "invent facts beyond what you are given."
        ),
        # No tools at all: reasoning only, over what the lead passes in.
        tools=[],
    ),
    "writer": AgentDefinition(
        description="Turns analysis into a short, well-structured report.",
        prompt=(
            "You write concise reports for a technical reader. Given findings and "
            "analysis, produce a short report: a one-line summary, the key points, "
            "and any stated limitations. No preamble, no filler."
        ),
        tools=[],
    ),
}

# `Agent` is the built-in delegation tool in claude-agent-sdk 0.2.x (the docs
# elsewhere call this "the Task tool"; the wire name observed from a live run is
# `Agent`, carrying `subagent_type`). `Task` is kept for forward/backward
# compatibility in case the name moves again. The read tools let the *subagents*
# do their work.
DELEGATION_TOOLS = ("Agent", "Task")
TEAM_TOOLS = ["Agent", "Task", "Grep", "Glob", "Read"]


@dataclass
class TeamResult:
    """The report, plus which subagents and tools were actually used."""

    report: str
    subagents_used: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str = "end_turn"


def _subagents_from(tool_calls) -> list[str]:
    """Which subagents the lead actually delegated to, in order.

    Delegations surface as ``Agent`` tool calls carrying a ``subagent_type``
    (verified against a live run; see :data:`DELEGATION_TOOLS`).
    """
    used: list[str] = []
    for call in tool_calls:
        if call.name in DELEGATION_TOOLS:
            name = call.input.get("subagent_type")
            if name:
                used.append(name)
    return used


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


async def run_team(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> TeamResult:
    """Answer `question` by delegating to the researcher/analyst/writer team.

    The lead decides who to delegate to; the roster in `TEAM` gives each
    subagent only the tools its role needs. The corpus gate covers the
    subagents' tool calls too, not just the lead's — verified live.
    """
    runner = runner or default_runner
    root = (corpus_dir or CORPUS_DIR).resolve()
    options = build_options(
        settings,
        system_prompt=LEAD_PROMPT,
        # DELIBERATELY EMPTY — an entry here auto-approves before the gate runs.
        # This matters more here than in a flat run: the gate has to cover the
        # *subagents'* tool calls too, not just the lead's.
        allowed_tools=[],
        tools=TEAM_TOOLS,
        agents=TEAM,
        # Where the team starts. NOT where it is allowed — that is the gate.
        cwd=str(root),
        permission_mode="default",
        can_use_tool=make_corpus_gate(root),
        # Multi-agent runs fan out, so they need more headroom than a flat run.
        # 12 was not enough: three delegations plus the lead's own read/write
        # turns intermittently hit the cap, and the run then returned an empty
        # report. Spend stays bounded by AGENT_MAX_BUDGET_USD, which is the cap
        # that actually protects the budget.
        max_turns=max(settings.agent_max_turns, 20),
    )
    result = await runner(question, options)
    return TeamResult(
        report=result.text,
        subagents_used=_subagents_from(result.tool_calls),
        tools_used=result.tool_names,
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
        stop_reason=outcome_of(result),
    )
