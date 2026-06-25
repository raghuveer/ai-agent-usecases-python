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
from typing import Callable

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
    """
    tools = tools if tools is not None else TOOLS
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(tools)},
        {"role": "user", "content": f"Task: {task}\n\nBegin."},
    ]
    steps: list[Step] = []
    nudged = False

    for _ in range(max_steps):
        reply = llm_call(messages)
        messages.append({"role": "assistant", "content": reply})

        final = parse_final_answer(reply)
        if final is not None:
            return ReactResult(answer=final, steps=steps, stopped_reason="final_answer")

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
            return ReactResult(
                answer=reply.strip(),
                steps=steps,
                stopped_reason="max_steps",
            )

        action, action_input = parsed
        observation = run_tool(tools, action, action_input)
        steps.append(
            Step(
                thought=parse_thought(reply),
                action=action,
                action_input=action_input,
                observation=observation,
            )
        )
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # Ran out of steps without a Final Answer.
    last_obs = steps[-1].observation if steps else ""
    return ReactResult(
        answer=last_obs,
        steps=steps,
        stopped_reason="max_steps",
    )
