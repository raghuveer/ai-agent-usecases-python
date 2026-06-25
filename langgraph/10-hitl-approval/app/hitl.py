# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langgraph). See langgraph/10-hitl-approval/README.md
"""Human-in-the-loop approval as a langgraph StateGraph — the showcase approach.

This is LangGraph's native territory: a workflow that **suspends at a checkpoint
for a human decision and resumes from exactly there**. The graph is three nodes:

    draft ──► review (interrupt) ──► execute ──► END

- **draft**: the LLM drafts the proposed high-risk action from the request.
- **review**: calls ``interrupt(...)`` — this SUSPENDS the graph and surfaces the
  proposed action to the caller. A checkpointer (``MemorySaver``) durably saves
  the entire graph state under ``thread_id == run_id``. Nothing executes yet.
- **execute**: runs only after the human resumes. The decision arrives as the
  return value of ``interrupt()`` via ``Command(resume=...)``; approved -> the
  action is marked sent (executed); not approved -> rejected.

``interrupt()`` + a checkpointer + ``Command(resume=...)`` is the clean native
mechanism: no hand-built store, no locks, no manual re-threading of state. The
LLM is injected, so unit tests pass a ``FakeListChatModel`` and run offline.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .llm import system_prefix

DRAFT_SYSTEM_PROMPT = (
    "You are an operations assistant. A human will review and approve your work "
    "before it takes effect. Given a request for a high-risk action, draft a "
    "single short, professional message that performs the action (for example a "
    "refund-approval note to a customer). Output ONLY the message text, no "
    "preamble, no explanation, at most 3 sentences."
)


class ApprovalState(TypedDict):
    request: str
    proposed_action: str
    approved: Optional[bool]
    feedback: Optional[str]
    status: str  # awaiting_approval | executed | rejected
    result: Optional[str]


def build_approval_graph(llm: BaseChatModel, settings=None):
    """Compile draft → review(interrupt) → execute → END with a checkpointer.

    The checkpointer (``MemorySaver``) is what makes ``interrupt()`` durable: the
    paused state is keyed by the thread id (which the app sets to ``run_id``).
    """

    def draft(state: ApprovalState) -> dict:
        messages = [
            SystemMessage(content=system_prefix(DRAFT_SYSTEM_PROMPT, settings)),
            HumanMessage(content=f"Request: {state['request']}\n\nDraft the action message:"),
        ]
        reply = llm.invoke(messages)
        text = (reply.content if hasattr(reply, "content") else str(reply)).strip()
        if not text:
            text = "(the model returned an empty draft)"
        return {"proposed_action": text, "status": "awaiting_approval"}

    def review(state: ApprovalState) -> dict:
        """Suspend here until a human resumes with their decision.

        ``interrupt`` returns the value passed to ``Command(resume=...)``. We pass
        a ``{"approved": bool, "feedback": str|None}`` dict.
        """
        decision = interrupt(
            {
                "proposed_action": state["proposed_action"],
                "message": "Approve this high-risk action?",
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        feedback = decision.get("feedback") if isinstance(decision, dict) else None
        return {"approved": approved, "feedback": feedback}

    def execute(state: ApprovalState) -> dict:
        if state.get("approved"):
            result = f"{state['proposed_action']}\n\n[ACTION EXECUTED / message sent]"
            return {"status": "executed", "result": result}
        return {"status": "rejected", "result": None}

    graph = StateGraph(ApprovalState)
    graph.add_node("draft", draft)
    graph.add_node("review", review)
    graph.add_node("execute", execute)
    graph.set_entry_point("draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", "execute")
    graph.add_edge("execute", END)

    # MemorySaver durably persists state at the interrupt so /resume can continue.
    return graph.compile(checkpointer=MemorySaver())
