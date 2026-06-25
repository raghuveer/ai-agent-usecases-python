# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langgraph). See langgraph/05-support-triage/README.md
"""Support triage modelled as a langgraph StateGraph.

This is the natural showcase for langgraph: triage is a *router*. The graph is

        classify
           |
   (conditional edge on intent)
       /    |    \\
  billing technical general      <- specialist responder nodes
       \\    |    /
        escalate                  <- sets escalate flag from confidence
           |
          END

``classify`` asks the LLM for ``{intent, confidence}`` JSON and writes both to
state. A conditional edge then routes to exactly one specialist node, each of
which generates the reply with its own system prompt. ``escalate`` finally sets
``escalate=true`` when confidence is below the threshold.

The LLM is injected (``FakeListChatModel`` in tests) so nothing hits the network.
A module-level dict gives per-session conversation memory; prior turns are fed
into both the classifier and the responder.
"""
from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import system_prompt
from .settings import Settings, get_settings

Intent = Literal["billing", "technical", "general"]
VALID_INTENTS: tuple[Intent, ...] = ("billing", "technical", "general")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
CLASSIFIER_SYSTEM = (
    "You are a strict support-ticket classification engine. "
    "You ONLY output a single JSON object and nothing else — no prose, no "
    "explanation, no apology."
)

# The schema/example and a primed ``JSON:`` suffix live in the human turn: small
# instruct models follow output-format rules far more reliably when the format is
# restated next to the input.
CLASSIFIER_USER_TEMPLATE = (
    "Classify the customer's latest message into exactly one intent.\n"
    "Intents:\n"
    "- billing: invoices, charges, refunds, payments, subscriptions, pricing.\n"
    "- technical: bugs, errors, outages, setup, how-to, broken features.\n"
    "- general: anything else (greetings, account questions, feedback).\n\n"
    "Output ONLY this JSON, no prose:\n"
    '{{"intent": "<billing|technical|general>", "confidence": <0.0-1.0>}}\n'
    "confidence is your certainty (0-1) in the chosen intent.\n\n"
    "{history}Customer message: {message}\n"
    "JSON:"
)

SPECIALIST_SYSTEM: dict[Intent, str] = {
    "billing": (
        "You are a billing-support specialist for Northwind Robotics. Help with "
        "invoices, charges, refunds and payments. Be concise, empathetic, and "
        "concrete about next steps. Never invent account-specific numbers."
    ),
    "technical": (
        "You are a technical-support specialist for Northwind Robotics. Help "
        "diagnose errors, outages and setup problems. Ask for the minimum info "
        "you need and give clear, step-by-step guidance."
    ),
    "general": (
        "You are a friendly general-support agent for Northwind Robotics. Answer "
        "the customer's question clearly, and route them to the right team if the "
        "request is really about billing or a technical issue."
    ),
}


# --------------------------------------------------------------------------- #
# In-memory session memory (a module-level dict is fine for the example)
# --------------------------------------------------------------------------- #
_SESSIONS: dict[str, list[dict[str, str]]] = {}


def get_history(session_id: str | None) -> list[dict[str, str]]:
    if not session_id:
        return []
    return list(_SESSIONS.get(session_id, []))


def append_turn(session_id: str | None, role: str, content: str) -> None:
    if not session_id:
        return
    _SESSIONS.setdefault(session_id, []).append({"role": role, "content": content})


def reset_memory() -> None:
    """Clear all session memory (used by tests for isolation)."""
    _SESSIONS.clear()


def _render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{t['role']}: {t['content']}" for t in history]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_classification(raw: str) -> tuple[Intent, float]:
    """Parse the classifier text into (intent, confidence), tolerantly."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "general", 0.0
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "general", 0.0

    intent = str(data.get("intent", "")).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "general"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class TriageState(TypedDict):
    message: str
    history: str
    intent: Intent
    confidence: float
    response: str
    escalate: bool


def route_intent(state: TriageState) -> str:
    """Conditional-edge function: pick the specialist node for the intent."""
    return state["intent"]


def build_triage_graph(llm: BaseChatModel, settings: Settings | None = None):
    """Compile classify -> (conditional) specialist -> escalate -> END."""
    settings = settings or get_settings()

    def classify(state: TriageState) -> dict:
        user = CLASSIFIER_USER_TEMPLATE.format(
            history=state["history"], message=state["message"]
        )
        messages = [
            SystemMessage(content=system_prompt(CLASSIFIER_SYSTEM, settings)),
            HumanMessage(content=user),
        ]
        raw = llm.invoke(messages).content
        intent, confidence = parse_classification(raw)
        return {"intent": intent, "confidence": confidence}

    def make_specialist(intent: Intent):
        def specialist(state: TriageState) -> dict:
            messages = [
                SystemMessage(content=system_prompt(SPECIALIST_SYSTEM[intent], settings)),
                HumanMessage(
                    content=state["history"] + f"Customer message: {state['message']}"
                ),
            ]
            return {"response": llm.invoke(messages).content}

        return specialist

    def finalize(state: TriageState) -> dict:
        # Escalate when the classifier's confidence is below the threshold.
        return {"escalate": state["confidence"] < settings.escalate_threshold}

    graph = StateGraph(TriageState)
    graph.add_node("classify", classify)
    graph.add_node("billing", make_specialist("billing"))
    graph.add_node("technical", make_specialist("technical"))
    graph.add_node("general", make_specialist("general"))
    # Node name must differ from the 'escalate' state key.
    graph.add_node("finalize", finalize)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {"billing": "billing", "technical": "technical", "general": "general"},
    )
    for node in ("billing", "technical", "general"):
        graph.add_edge(node, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


# --------------------------------------------------------------------------- #
# Full triage pass (wires session memory around a graph invocation)
# --------------------------------------------------------------------------- #
def run_triage(graph, message: str, *, session_id: str | None) -> dict:
    """Invoke the compiled graph with session history, then persist the turn."""
    history = _render_history(get_history(session_id))
    out = graph.invoke(
        {
            "message": message,
            "history": history,
            "intent": "general",
            "confidence": 0.0,
            "response": "",
            "escalate": False,
        }
    )
    append_turn(session_id, "user", message)
    append_turn(session_id, "assistant", out["response"])
    return {
        "intent": out["intent"],
        "confidence": out["confidence"],
        "response": out["response"],
        "escalate": out["escalate"],
    }
