# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""Multi-agent orchestration via the SDK's native subagents.

`ClaudeAgentOptions.agents` takes a dict of :class:`AgentDefinition`s. The lead
agent then delegates to them with the built-in `Task` tool, and the SDK handles
spawning, isolating, and collecting each one:

    lead ──► Task(researcher) ──► findings ┐
         ──► Task(analyst)   ──► analysis  ├──► lead writes the report
         ──► Task(writer)    ──► prose     ┘

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

from claude_agent_sdk import AgentDefinition

from .agent import Runner, build_options, default_runner
from .settings import Settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data"

LEAD_PROMPT = """You lead a small research team working from a local document corpus.

For the user's question:
1. Delegate corpus gathering to the `researcher` subagent.
2. Delegate interpretation of those findings to the `analyst` subagent.
3. Delegate the final write-up to the `writer` subagent.
4. Return the writer's report as your final answer.

Use the Task tool to delegate. Do not read the corpus yourself — that is the
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

# `Task` is the built-in delegation tool; the read tools let the *subagents* work.
TEAM_TOOLS = ["Task", "Grep", "Glob", "Read"]


@dataclass
class TeamResult:
    report: str
    subagents_used: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float


def _subagents_from(tool_calls) -> list[str]:
    """Which subagents the lead actually delegated to, in order.

    Delegations surface as `Task` tool calls carrying a `subagent_type`.
    """
    used: list[str] = []
    for call in tool_calls:
        if call.name == "Task":
            name = call.input.get("subagent_type")
            if name:
                used.append(name)
    return used


async def run_team(
    question: str,
    settings: Settings,
    runner: Runner | None = None,
    corpus_dir: Path | None = None,
) -> TeamResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=LEAD_PROMPT,
        allowed_tools=TEAM_TOOLS,
        tools=TEAM_TOOLS,
        agents=TEAM,
        cwd=str(corpus_dir or CORPUS_DIR),
        # Multi-agent runs fan out, so they need more headroom than a flat run.
        max_turns=max(settings.agent_max_turns, 12),
    )
    result = await runner(question, options)
    return TeamResult(
        report=result.text,
        subagents_used=_subagents_from(result.tool_calls),
        tools_used=result.tool_names,
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
    )
