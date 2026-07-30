# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langgraph). See langgraph/08-autonomous-react/README.md
"""UC8 ReAct agent as a langgraph StateGraph — the showcase approach.

This is LangGraph's native territory: an autonomous agent that *loops until
done*. The graph makes the ReAct cycle explicit as a set of nodes joined by a
**conditional edge** that either loops back or terminates:

    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
  reason ──► (route) ──tool──► act ──► observe ───────────┘
    │
    └──final / max_steps──► END

- **reason**: call the LLM with the running transcript; it emits either an
  ``Action``/``Action Input`` (text ReAct protocol) or a ``Final Answer``.
- **route** (conditional edge): if the model produced a Final Answer, or we hit
  ``max_steps``, go to END; otherwise go to ``act``.
- **act**: dispatch the parsed tool from the ``TOOLS`` registry.
- **observe**: append ``Observation: <result>`` to the transcript and bump the
  step counter, then the static edge loops back to ``reason``.

We use the TEXT ReAct protocol (not provider-native function-calling, which is
inconsistent on this gateway). The LLM is injected, so unit tests pass a
``FakeListChatModel`` and the whole cycle runs offline.
"""
from __future__ import annotations

import re
from typing import Annotated, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .llm import stop_sequences, system_prefix, truncate_at_stop
from .tools import TOOLS, UnsafeExpression

ToolRegistry = dict  # name -> (callable, description)


# --------------------------------------------------------------------------- #
# Typed graph state — the transcript lives here.
# --------------------------------------------------------------------------- #
class ReactState(TypedDict):
    task: str
    max_steps: int
    # The conversation transcript; add_messages appends across the loop.
    messages: Annotated[list[BaseMessage], add_messages]
    # Structured steps collected for the API response.
    steps: list[dict]
    # Set when the model emits a Final Answer.
    answer: Optional[str]
    stopped_reason: Optional[str]
    # Pending tool call parsed in `reason`, consumed by `act`.
    pending_action: Optional[str]
    pending_input: Optional[str]
    pending_thought: Optional[str]
    # Last observation, threaded by `observe`.
    last_observation: Optional[str]


def build_system_prompt(tools: ToolRegistry) -> str:
    tool_lines = "\n".join(f"- {name}: {desc}" for name, (_, desc) in tools.items())
    names = ", ".join(tools.keys())
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
    return action, (m.group(1).strip() if m else "")


def parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def run_tool(tools: ToolRegistry, action: str, action_input: str) -> str:
    entry = tools.get(action)
    if entry is None:
        valid = ", ".join(tools.keys())
        return f"Error: unknown tool '{action}'. Valid tools: {valid}."
    func = entry[0]
    try:
        return str(func(action_input))
    except UnsafeExpression as exc:
        return f"Error: unsafe or invalid expression ({exc})."
    except Exception as exc:
        return f"Error: {exc}"


# --------------------------------------------------------------------------- #
# Graph construction — the ReAct cycle.
# --------------------------------------------------------------------------- #
def build_react_graph(llm: BaseChatModel, tools: ToolRegistry | None = None):
    """Compile the reason → act → observe → (loop | END) StateGraph."""
    tools = tools if tools is not None else TOOLS

    def reason(state: ReactState) -> dict:
        """Call the LLM with the transcript; parse a tool call or final answer."""
        # Stop after the model's Action so it can't hallucinate the Observation.
        # `stop` is sent only where the endpoint honours it, so the cut is also
        # applied locally — before the turn enters the message history.
        reply = llm.invoke(state["messages"], stop=stop_sequences(llm))
        text = reply.content if isinstance(reply, AIMessage) else str(reply)
        text = truncate_at_stop(text)

        final = parse_final_answer(text)
        if final is not None:
            return {
                "messages": [AIMessage(content=text)],
                "answer": final,
                "stopped_reason": "final_answer",
                "pending_action": None,
            }

        parsed = parse_action(text)
        if parsed is None:
            # No Action and no Final Answer: stop deterministically.
            return {
                "messages": [AIMessage(content=text)],
                "answer": text.strip(),
                "stopped_reason": "max_steps",
                "pending_action": None,
            }

        action, action_input = parsed
        return {
            "messages": [AIMessage(content=text)],
            "pending_action": action,
            "pending_input": action_input,
            "pending_thought": parse_thought(text),
        }

    def act(state: ReactState) -> dict:
        """Dispatch the pending tool call from the registry."""
        observation = run_tool(tools, state["pending_action"], state["pending_input"] or "")
        return {"last_observation": observation}

    def observe(state: ReactState) -> dict:
        """Record the step and thread the Observation back into the transcript."""
        step = {
            "thought": state.get("pending_thought") or "",
            "action": state["pending_action"],
            "action_input": state.get("pending_input") or "",
            "observation": state["last_observation"] or "",
        }
        return {
            "steps": state["steps"] + [step],
            "messages": [HumanMessage(content=f"Observation: {state['last_observation']}")],
            "pending_action": None,
            "pending_input": None,
            "pending_thought": None,
            "last_observation": None,
        }

    def route(state: ReactState) -> str:
        """Conditional edge: END on final answer / step-cap, else loop to act."""
        if state.get("stopped_reason") == "final_answer":
            return "end"
        if len(state["steps"]) >= state["max_steps"]:
            # Cap reached without a final answer.
            return "end"
        return "act"

    graph = StateGraph(ReactState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("observe", observe)

    graph.set_entry_point("reason")
    # reason decides: keep looping (act) or stop (END).
    graph.add_conditional_edges("reason", route, {"act": "act", "end": END})
    graph.add_edge("act", "observe")
    # observe loops back to reason — this static back-edge is the cycle.
    graph.add_edge("observe", "reason")
    return graph.compile()


def run_react(
    task: str,
    *,
    llm: BaseChatModel,
    tools: ToolRegistry | None = None,
    max_steps: int = 6,
    callbacks: list | None = None,
) -> dict:
    """Run the compiled graph and return a normalised result dict.

    Returns ``{"answer", "steps", "stopped_reason"}``.

    ``callbacks`` go in the graph's run config rather than on the model, because
    that is the only place LangGraph reports which *node* is executing — which is
    what makes the route observable (see app/trace.py).
    """
    tools = tools if tools is not None else TOOLS
    graph = build_react_graph(llm, tools)
    init: ReactState = {
        "task": task,
        "max_steps": max_steps,
        "messages": [
            SystemMessage(content=build_system_prompt(tools)),
            HumanMessage(content=f"Task: {task}\n\nBegin."),
        ],
        "steps": [],
        "answer": None,
        "stopped_reason": None,
        "pending_action": None,
        "pending_input": None,
        "pending_thought": None,
        "last_observation": None,
    }
    # recursion_limit must allow reason+act+observe per step (3 nodes/cycle).
    config: dict = {"recursion_limit": max_steps * 3 + 5}
    if callbacks:
        config["callbacks"] = callbacks
    out = graph.invoke(init, config=config)

    steps = out.get("steps", [])
    stopped_reason = out.get("stopped_reason")
    if stopped_reason is None:
        stopped_reason = "max_steps"
    answer = out.get("answer")
    if answer is None:
        answer = steps[-1]["observation"] if steps else ""
    return {"answer": answer, "steps": steps, "stopped_reason": stopped_reason}
