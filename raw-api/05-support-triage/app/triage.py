# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (raw-api). See raw-api/05-support-triage/README.md
"""Hand-written support triage: classify -> route -> respond -> escalate.

This is the raw-api point: nothing is hidden behind a framework. We

1. ask the LLM to classify the message into an intent + confidence (as JSON),
2. route to a per-intent specialist system prompt,
3. ask the LLM to write the reply with that specialist prompt,
4. escalate when the classifier's confidence falls below a threshold.

A tiny module-level dict gives per-session conversation memory; the prior turns
of a session are fed back into both the classifier and the responder so the
example shows multi-turn context. The classifier call and the responder call are
both expressed as a single injectable ``LLMCall`` so unit tests swap in fakes and
never touch the network.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Literal

Intent = Literal["billing", "technical", "general"]
VALID_INTENTS: tuple[Intent, ...] = ("billing", "technical", "general")

# An LLM call is ``(system_prompt, user_prompt) -> text``. Injectable for tests.
LLMCall = Callable[[str, str], str]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
CLASSIFIER_SYSTEM = (
    "You are a strict support-ticket classification engine. "
    "You ONLY output a single JSON object and nothing else — no prose, no "
    "explanation, no apology."
)

# The schema, intent definitions and a primed ``JSON:`` suffix all live in the
# user turn: small instruct models follow output-format rules far more reliably
# when the format is restated next to the input with a concrete example.
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
# Maps session_id -> list of {"role": "user"|"assistant", "content": str}.
_SESSIONS: dict[str, list[dict[str, str]]] = {}


def get_history(session_id: str | None) -> list[dict[str, str]]:
    """Return the stored turns for a session (empty if none / no id)."""
    if not session_id:
        return []
    return list(_SESSIONS.get(session_id, []))


def append_turn(session_id: str | None, role: str, content: str) -> None:
    """Append a turn to a session's memory (no-op without a session id)."""
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
# Classification
# --------------------------------------------------------------------------- #
def parse_classification(raw: str) -> tuple[Intent, float]:
    """Parse the classifier's JSON reply into (intent, confidence).

    Tolerant of fenced code blocks and surrounding prose: we pull the first
    ``{...}`` object out. Unknown intents fall back to ``general`` with low
    confidence so the request still escalates rather than crashes.
    """
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
    # Clamp to [0, 1].
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence  # type: ignore[return-value]


def classify(
    message: str,
    *,
    llm_call: LLMCall,
    history: list[dict[str, str]] | None = None,
) -> tuple[Intent, float]:
    """Classify a message into (intent, confidence) via the LLM."""
    history = history or []
    user_prompt = CLASSIFIER_USER_TEMPLATE.format(
        history=_render_history(history), message=message
    )
    raw = llm_call(CLASSIFIER_SYSTEM, user_prompt)
    return parse_classification(raw)


# --------------------------------------------------------------------------- #
# Responding
# --------------------------------------------------------------------------- #
def respond(
    message: str,
    intent: Intent,
    *,
    llm_call: LLMCall,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generate the specialist reply for the routed intent."""
    history = history or []
    system = SPECIALIST_SYSTEM[intent]
    user_prompt = _render_history(history) + f"Customer message: {message}"
    return llm_call(system, user_prompt)


# --------------------------------------------------------------------------- #
# Full triage pass
# --------------------------------------------------------------------------- #
def triage(
    message: str,
    *,
    session_id: str | None,
    classify_call: LLMCall,
    respond_call: LLMCall,
    escalate_threshold: float,
) -> dict:
    """Full pass: classify -> route -> respond -> escalate, with session memory.

    ``classify_call`` and ``respond_call`` are separate injectable LLM calls so
    tests can mock the classifier and the responder independently.
    """
    history = get_history(session_id)

    intent, confidence = classify(message, llm_call=classify_call, history=history)
    response = respond(message, intent, llm_call=respond_call, history=history)
    escalate = confidence < escalate_threshold

    # Persist this turn for subsequent requests in the same session.
    append_turn(session_id, "user", message)
    append_turn(session_id, "assistant", response)

    return {
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "escalate": escalate,
    }
