# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langchain). See langchain/05-support-triage/README.md
"""Support triage with langchain: classify -> route -> respond -> escalate.

Two small LCEL chains do the work:

- the **classifier chain** = ``prompt | llm | StrOutputParser`` whose text we
  parse into an ``{intent, confidence}`` pair;
- a **responder chain** per intent = ``specialist_prompt | llm | StrOutputParser``.

Routing is a plain dict lookup on the classified intent. A module-level dict
gives per-session conversation memory; prior turns are fed back into both chains
via a ``{history}`` prompt variable so the example shows multi-turn context.

The LLM is injected (``FakeListChatModel`` in tests) so nothing hits the network.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm import system_prompt

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
# restated next to the input. Braces are doubled for ChatPromptTemplate.
CLASSIFIER_USER = (
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

USER_TEMPLATE = "{history}Customer message: {message}"


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


def render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{t['role']}: {t['content']}" for t in history]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------- #
# Parsing the classifier output
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
# Chains
# --------------------------------------------------------------------------- #
def build_classifier_chain(llm: BaseChatModel):
    """LCEL chain: {message, history} -> raw classifier text."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt(CLASSIFIER_SYSTEM)), ("human", CLASSIFIER_USER)]
    )
    return prompt | llm | StrOutputParser()


def build_responder_chain(llm: BaseChatModel, intent: Intent):
    """LCEL chain: {message, history} -> specialist reply text."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt(SPECIALIST_SYSTEM[intent])), ("human", USER_TEMPLATE)]
    )
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------- #
# Full triage pass
# --------------------------------------------------------------------------- #
def triage(
    message: str,
    *,
    session_id: str | None,
    llm: BaseChatModel,
    escalate_threshold: float,
) -> dict:
    """Full pass: classify -> route -> respond -> escalate, with session memory."""
    history = get_history(session_id)
    history_text = render_history(history)

    raw = build_classifier_chain(llm).invoke(
        {"message": message, "history": history_text}
    )
    intent, confidence = parse_classification(raw)

    response = build_responder_chain(llm, intent).invoke(
        {"message": message, "history": history_text}
    )
    escalate = confidence < escalate_threshold

    append_turn(session_id, "user", message)
    append_turn(session_id, "assistant", response)

    return {
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "escalate": escalate,
    }
