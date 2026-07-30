# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langchain). See langchain/08-autonomous-react/README.md
"""Text-based ReAct agent assembled from LangChain primitives (langchain approach).

The spec requires a TEXT ReAct protocol (not provider-native function-calling,
which is inconsistent on this gateway). So rather than ``create_react_agent`` +
``AgentExecutor`` (which lean on native tool-calling), we build the agent from
LangChain ``Tool`` objects and a ``BaseChatModel``, and drive the same strict
text format:

    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <single-line input>

...or, when finished:

    Thought: <reasoning>
    Final Answer: <answer>

The loop invokes the chat model with a LangChain message list, parses the last
`Action`/`Action Input`, calls the matching ``Tool``, and threads the result
back as an ``Observation:``. The chat model is injected, so unit tests pass a
``FakeListChatModel`` and run offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool

from .llm import (
    ThinkFilter,
    stop_sequences,
    strip_thinking,
    system_prefix,
    truncate_at_stop,
)


@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class ReactResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_steps"


def build_system_prompt(tools: list[Tool]) -> str:
    """System prompt describing the tools and the exact ReAct text format."""
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    names = ", ".join(t.name for t in tools)
    text = (
        "You are a tool-using agent that solves a task step by step.\n"
        "You have access to these tools:\n"
        f"{tool_lines}\n\n"
        "Work in a loop. On each turn output EXACTLY this format to call a tool:\n"
        "Thought: <your reasoning>\n"
        "Action: <one of: " + names + ">\n"
        "Arguments: <single-line input for the tool>\n\n"
        "Then STOP and wait. You will receive a line:\n"
        "Observation: <tool result>\n\n"
        "When you have enough information, output instead:\n"
        "Thought: <your reasoning>\n"
        "Final Answer: <the final answer>\n\n"
        "Rules: output only ONE Action per turn, then stop — do NOT write the "
        "Observation yourself. Use the calculator for any arithmetic. Use search "
        "to look up Northwind facts. Do not invent facts."
    )
    return system_prefix(text)


# --- Parsing ---------------------------------------------------------------- #
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)", re.IGNORECASE)
# The model emits "Arguments:" (not "Action Input:" — the gateway's PII filter
# masks the phrase "Action Input" as <PERSON>). Tolerate both for robustness.
_INPUT_RE = re.compile(r"(?:Arguments|Action Input):\s*([^\n]+)", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"Thought:\s*([^\n]+)", re.IGNORECASE)


def parse_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def parse_action(text: str) -> tuple[str, str] | None:
    actions = list(_ACTION_RE.finditer(text))
    if not actions:
        return None
    action = actions[-1].group(1).strip()
    tail = text[actions[-1].end():]
    m = _INPUT_RE.search(tail)
    action_input = m.group(1).strip() if m else ""
    return action, action_input


def parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {t.name: t for t in tools}


def run_tool(tools: list[Tool], action: str, action_input: str) -> str:
    """Dispatch to a LangChain Tool by name, returning its observation string."""
    tmap = _tool_map(tools)
    tool = tmap.get(action)
    if tool is None:
        valid = ", ".join(tmap.keys())
        return f"Error: unknown tool '{action}'. Valid tools: {valid}."
    try:
        return str(tool.invoke(action_input))
    except Exception as exc:  # keep the loop alive; surface the error
        return f"Error: {exc}"


# --- Loop ------------------------------------------------------------------- #
def run_react(
    task: str,
    *,
    llm: BaseChatModel,
    tools: list[Tool],
    max_steps: int = 6,
) -> ReactResult:
    """Run the LangChain-tool ReAct loop until Final Answer or max_steps.

    Drains :func:`iter_react`, so the blocking and streaming paths cannot drift
    apart — there is one loop, exposed two ways.
    """
    result: ReactResult | None = None
    for event in iter_react(task, llm=llm, tools=tools, max_steps=max_steps):
        if event["type"] == "final":
            result = event["result"]
    assert result is not None, "iter_react always ends with a final event"
    return result


def iter_react(
    task: str,
    *,
    llm: BaseChatModel,
    tools: list[Tool],
    max_steps: int = 6,
    stream: bool = False,
) -> Iterator[dict]:
    """The ReAct loop as a stream of events. See docs/streaming.md.

    With ``stream=True`` the model is driven by ``llm.stream()`` — LangChain's
    own incremental API, which yields ``AIMessageChunk``s. That is the whole
    difference from the raw-api version: no SSE parsing, no delta bookkeeping,
    just an iterator the framework hands you.
    """
    messages = [
        SystemMessage(content=build_system_prompt(tools)),
        HumanMessage(content=f"Task: {task}\n\nBegin."),
    ]
    steps: list[Step] = []
    nudged = False

    for _ in range(max_steps):
        # Stop after the model's Action so it can't hallucinate the Observation.
        # `stop` is sent only where the endpoint honours it, so the cut is also
        # applied locally — before the turn enters the transcript.
        if stream:
            think = ThinkFilter()
            pieces: list[str] = []
            for chunk in llm.stream(messages, stop=stop_sequences(llm)):
                raw = chunk.content if hasattr(chunk, "content") else str(chunk)
                visible = think.feed(raw if isinstance(raw, str) else str(raw))
                if visible:
                    pieces.append(visible)
                    yield {"type": "token", "text": visible}
            tail = think.flush()
            if tail:
                pieces.append(tail)
                yield {"type": "token", "text": tail}
            text = truncate_at_stop("".join(pieces).strip())
        else:
            reply = llm.invoke(messages, stop=stop_sequences(llm))
            text = strip_thinking(
                reply.content if isinstance(reply, AIMessage) else str(reply)
            )
            text = truncate_at_stop(text)
        messages.append(AIMessage(content=text))

        thought = parse_thought(text)
        if thought:
            yield {"type": "thought", "text": thought}

        final = parse_final_answer(text)
        if final is not None:
            yield {
                "type": "final",
                "result": ReactResult(
                    answer=final, steps=steps, stopped_reason="final_answer"
                ),
            }
            return

        parsed = parse_action(text)
        if parsed is None:
            if not nudged:
                nudged = True
                messages.append(HumanMessage(content=(
                    "Please respond using the required format: either an Action "
                    "with Action Input, or a Final Answer."
                )))
                continue
            yield {
                "type": "final",
                "result": ReactResult(
                    answer=text.strip(), steps=steps, stopped_reason="max_steps"
                ),
            }
            return

        action, action_input = parsed
        observation = run_tool(tools, action, action_input)
        step = Step(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        steps.append(step)
        yield {"type": "step", "step": step}
        messages.append(HumanMessage(content=f"Observation: {observation}"))

    last_obs = steps[-1].observation if steps else ""
    yield {
        "type": "final",
        "result": ReactResult(
            answer=last_obs, steps=steps, stopped_reason="max_steps"
        ),
    }
