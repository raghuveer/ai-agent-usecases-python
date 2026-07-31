# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC08 Autonomous ReAct (claude-agent-sdk). See claude-agent-sdk/08-autonomous-react/README.md
"""Autonomous reason-act loop — except there is no loop to write.

The Agent SDK *is* a ReAct loop. You register tools; the model decides which to
call and when to stop; `max_turns` bounds it. This module therefore contains only
tools and a prompt.

What that removes is not cosmetic. The text-ReAct implementations in this repo
needed all of the following, and none of it exists here:

* a parser for `Thought:` / `Action:` / `Arguments:` lines
* `stop=["Observation:"]`, or the model hallucinates its own tool results
* a rename of the `Action Input` field, because the gateway's PII redaction
  masked that literal string to `<PERSON>` and broke parsing
* a hand-written step ceiling and "never reached Final Answer" fallback

Tool calls here are structured protocol messages, not prose the model must be
coaxed into formatting correctly — so there is no text for a redactor to mangle
and nothing to parse.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import Settings

# A tiny fixed "warehouse" so runs are deterministic and offline.
METRICS: dict[str, float] = {
    "monthly_revenue_usd": 128_400.0,
    "monthly_costs_usd": 91_250.0,
    "active_customers": 1_820.0,
    "churned_customers": 96.0,
    "support_tickets": 412.0,
}

# Arithmetic-only evaluator. `eval()` on model-authored text would be a remote
# code execution hole; this walks the AST and permits nothing but numbers and
# the five operators below.
_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression, rejecting everything else."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric literals are allowed")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    return _eval(ast.parse(expression, mode="eval").body)


@tool("lookup_metric", "Look up a business metric by name.", {"name": str})
async def lookup_metric(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("name", "")).strip()
    if name not in METRICS:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Unknown metric '{name}'. Available: {', '.join(sorted(METRICS))}.",
                }
            ],
            "is_error": True,
        }
    return {"content": [{"type": "text", "text": f"{name} = {METRICS[name]}"}]}


@tool(
    "calculate",
    "Evaluate an arithmetic expression, e.g. '(128400 - 91250) / 128400'.",
    {"expression": str},
)
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        value = safe_eval(str(args.get("expression", "")))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        # Returned as an error tool result so the agent can correct itself
        # rather than the whole run dying.
        return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}
    return {"content": [{"type": "text", "text": str(value)}]}


SYSTEM_PROMPT = """You answer questions about business metrics.

You cannot see the metric values directly — use the lookup_metric tool to fetch
each one you need, and the calculate tool for any arithmetic. Never do arithmetic
in your head and never guess a metric value.

When you have the answer, state it plainly with the numbers you used."""

REACT_TOOLS = ["mcp__metrics__lookup_metric", "mcp__metrics__calculate"]


def build_metrics_server():
    return create_sdk_mcp_server(
        name="metrics", version="1.0.0", tools=[lookup_metric, calculate]
    )


@dataclass
class Step:
    tool: str
    input: dict[str, Any]


@dataclass
class ReactResult:
    answer: str
    trace: list[Step]
    num_turns: int
    cost_usd: float
    hit_turn_limit: bool
    stop_reason: str = "end_turn"


async def run_react(
    question: str, settings: Settings, runner: Runner | None = None
) -> ReactResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=REACT_TOOLS,
        mcp_servers={"metrics": build_metrics_server()},
        max_turns=max(settings.agent_max_turns, 8),
    )
    result = await runner(question, options)

    return ReactResult(
        answer=result.text,
        # Strip the mcp__server__ prefix so the trace reads as tool names.
        trace=[
            Step(tool=c.name.split("__")[-1], input=c.input) for c in result.tool_calls
        ],
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
        stop_reason=outcome_of(result),
        hit_turn_limit=result.stop_reason == "max_turns",
    )
