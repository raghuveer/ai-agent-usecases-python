# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""Hand-written text-based ReAct loop (raw-api approach).

This is the raw-api point: the agentic loop is plain Python, nothing hidden
behind a framework. We drive ANY chat model with a TEXT protocol (not provider
-native function-calling, which is inconsistent on this gateway):

    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <single-line input>

...or, when finished:

    Thought: <reasoning>
    Final Answer: <answer>

Each iteration: call the LLM -> parse the last Action/Action Input -> run the
tool -> append ``Observation: <result>`` -> loop. We stop on ``Final Answer:``
or when ``max_steps`` is hit. Parsing is defensive: if the model emits neither
an Action nor a Final Answer, we nudge it once, then stop.

The LLM call is injected as ``llm_call(messages) -> str`` so unit tests script
a sequence of replies and never touch the network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .tools import TOOLS, UnsafeExpression

# An LLM call maps a message list to the assistant's reply text. Injectable.
LLMCall = Callable[[list[dict]], str]

ToolRegistry = dict  # name -> (callable, description)


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


def build_system_prompt(tools: ToolRegistry) -> str:
    """System prompt describing the tools and the exact ReAct text format."""
    tool_lines = "\n".join(
        f"- {name}: {desc}" for name, (_, desc) in tools.items()
    )
    names = ", ".join(tools.keys())
    return (
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


# --- Parsing ---------------------------------------------------------------- #
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)", re.IGNORECASE)
# The model emits "Arguments:" (not "Action Input:" — the gateway's PII filter
# masks the phrase "Action Input" as <PERSON>). Tolerate both for robustness.
_INPUT_RE = re.compile(r"(?:Arguments|Action Input):\s*([^\n]+)", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"Thought:\s*([^\n]+)", re.IGNORECASE)


def parse_final_answer(text: str) -> str | None:
    """Return the Final Answer text if present, else None."""
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def parse_action(text: str) -> tuple[str, str] | None:
    """Return (action, action_input) for the LAST action block, else None."""
    actions = list(_ACTION_RE.finditer(text))
    if not actions:
        return None
    action = actions[-1].group(1).strip()
    # Take the Action Input that follows the last Action, if any.
    tail = text[actions[-1].end():]
    m = _INPUT_RE.search(tail)
    action_input = m.group(1).strip() if m else ""
    return action, action_input


def parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def run_tool(tools: ToolRegistry, action: str, action_input: str) -> str:
    """Dispatch to a registered tool, returning its observation string."""
    entry = tools.get(action)
    if entry is None:
        valid = ", ".join(tools.keys())
        return f"Error: unknown tool '{action}'. Valid tools: {valid}."
    func = entry[0]
    try:
        return str(func(action_input))
    except UnsafeExpression as exc:
        return f"Error: unsafe or invalid expression ({exc})."
    except Exception as exc:  # keep the loop alive; surface the error to the model
        return f"Error: {exc}"


# --- Loop ------------------------------------------------------------------- #
def run_react(
    task: str,
    *,
    llm_call: LLMCall,
    tools: ToolRegistry | None = None,
    max_steps: int = 6,
) -> ReactResult:
    """Run the ReAct loop until Final Answer or max_steps.

    ``llm_call(messages)`` returns the assistant reply text for a message list.

    Implemented by draining :func:`iter_react`, so the blocking and streaming
    paths cannot drift apart — there is one loop, exposed two ways.
    """
    result: ReactResult | None = None
    for event in iter_react(
        task, llm_call=llm_call, tools=tools, max_steps=max_steps
    ):
        if event["type"] == "final":
            result = event["result"]
    assert result is not None, "iter_react always ends with a final event"
    return result


def iter_react(
    task: str,
    *,
    llm_call: LLMCall | None = None,
    llm_stream: Callable[[list[dict]], Iterator[str]] | None = None,
    tools: ToolRegistry | None = None,
    max_steps: int = 6,
) -> Iterator[dict]:
    """The ReAct loop as a stream of events.

    Yields, in order as they happen:

    * ``{"type": "token", "text": …}`` — only when ``llm_stream`` is given
    * ``{"type": "thought", "text": …}`` — the model's reasoning for this turn
    * ``{"type": "step", "step": Step}`` — a tool ran, with its observation
    * ``{"type": "final", "result": ReactResult}`` — always last

    Exactly one of ``llm_call`` / ``llm_stream`` is used; the streaming variant
    assembles the same reply text from its chunks, so parsing is identical
    either way.
    """
    if llm_call is None and llm_stream is None:
        raise ValueError("pass llm_call or llm_stream")
    tools = tools if tools is not None else TOOLS
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": f"Task: {task}\n\nBegin."},
    ]
    steps: list[Step] = []
    nudged = False

    for _ in range(max_steps):
        if llm_stream is not None:
            pieces: list[str] = []
            for piece in llm_stream(messages):
                pieces.append(piece)
                yield {"type": "token", "text": piece}
            reply = "".join(pieces).strip()
        else:
            reply = llm_call(messages)
        messages.append({"role": "assistant", "content": reply})

        thought = parse_thought(reply)
        if thought:
            yield {"type": "thought", "text": thought}

        final = parse_final_answer(reply)
        if final is not None:
            yield {
                "type": "final",
                "result": ReactResult(
                    answer=final, steps=steps, stopped_reason="final_answer"
                ),
            }
            return

        parsed = parse_action(reply)
        if parsed is None:
            # No Action and no Final Answer. Nudge once, then give up.
            if not nudged:
                nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Please respond using the required format: either an "
                        "Action with Action Input, or a Final Answer."
                    ),
                })
                continue
            yield {
                "type": "final",
                "result": ReactResult(
                    answer=reply.strip(), steps=steps, stopped_reason="max_steps"
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
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # Ran out of steps without a Final Answer.
    last_obs = steps[-1].observation if steps else ""
    yield {
        "type": "final",
        "result": ReactResult(
            answer=last_obs, steps=steps, stopped_reason="max_steps"
        ),
    }
