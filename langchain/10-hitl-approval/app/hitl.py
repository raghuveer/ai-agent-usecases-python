# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langchain). See langchain/10-hitl-approval/README.md
"""Human-in-the-loop approval core (langchain approach).

LangChain has no first-class ``interrupt()``: a chain (LCEL pipeline) is meant to
run start-to-finish. So a human checkpoint is a **manual pause/resume workaround**
built *around* the chain:

- The draft chain (``prompt | llm | StrOutputParser``) runs to completion in
  ``/run``, producing the proposed action.
- We then **explicitly stop** and stash the run state in a ``RunRegistry`` keyed
  by ``run_id``. The chain is not "paused" — there is no graph to suspend — we
  just hold the result and decline to execute.
- ``/resume`` looks the run up and runs the (trivial) execute step. We remove the
  run on a terminal decision so it can't resume twice.

Contrast with ``langgraph/10-hitl-approval``, where ``interrupt()`` + a
checkpointer suspends and resumes the actual workflow. Here the "pause" is
application-level bookkeeping the developer owns.

The LLM is injected, so unit tests pass a ``FakeListChatModel`` and run offline.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm import strip_thinking_stream, system_prefix

DRAFT_SYSTEM_PROMPT = (
    "You are an operations assistant. A human will review and approve your work "
    "before it takes effect. Given a request for a high-risk action, draft a "
    "single short, professional message that performs the action (for example a "
    "refund-approval note to a customer). Output ONLY the message text, no "
    "preamble, no explanation, at most 3 sentences."
)


def build_draft_chain(llm: BaseChatModel, settings=None):
    """LCEL chain: prompt | llm | StrOutputParser -> the proposed action text."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prefix(DRAFT_SYSTEM_PROMPT, settings)),
            ("human", "Request: {request}\n\nDraft the action message:"),
        ]
    )
    return prompt | llm | StrOutputParser() | strip_thinking_stream


# --------------------------------------------------------------------------- #
# Manual pause registry — the workaround for the missing native interrupt.
# --------------------------------------------------------------------------- #
@dataclass
class PausedRun:
    run_id: str
    request: str
    proposed_action: str
    status: str = "awaiting_approval"


@dataclass
class RunRegistry:
    """Thread-safe store of paused runs keyed by run_id (app-level bookkeeping)."""

    _runs: dict[str, PausedRun] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, run: PausedRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get(self, run_id: str) -> Optional[PausedRun]:
        with self._lock:
            return self._runs.get(run_id)

    def pop(self, run_id: str) -> Optional[PausedRun]:
        with self._lock:
            return self._runs.pop(run_id, None)


class UnknownRunError(KeyError):
    """Raised when /resume is given a run_id that is not (or no longer) paused."""


def iter_start_run(request: str, *, chain, registry: RunRegistry) -> Iterator[dict]:
    """Draft with ``chain.stream()``, then park. See docs/streaming.md.

    Ends at the approval gate: the last frame is ``awaiting_approval`` and the
    caller's connection closes. Holding a socket open across a human decision is
    the fragility a durable approval workflow exists to avoid.

    Note the chain must end in ``strip_thinking_stream`` rather than
    ``strip_thinking`` — a plain function in an LCEL pipe becomes a
    ``RunnableLambda``, which buffers its whole input and silently turns this
    stream back into one chunk (TRACKING finding 17).
    """
    pieces: list[str] = []
    for piece in chain.stream({"request": request}):
        text = piece if isinstance(piece, str) else str(piece)
        if text:
            pieces.append(text)
            yield {"type": "token", "text": text}

    proposed = "".join(pieces).strip() or "(the model returned an empty draft)"
    run = PausedRun(run_id=uuid.uuid4().hex, request=request, proposed_action=proposed)
    registry.save(run)
    yield {
        "type": "awaiting_approval",
        "run_id": run.run_id,
        "proposed_action": run.proposed_action,
    }


def iter_resume_run(
    run_id: str,
    *,
    approved: bool,
    registry: RunRegistry,
    feedback: str | None = None,
) -> Iterator[dict]:
    """Resume as a second, short stream: the decision, then the outcome.

    No ``token`` frames — approving executes the text the human already read.
    """
    yield {"type": "decision", "approved": approved}
    outcome = resume_run(
        run_id, approved=approved, registry=registry, feedback=feedback
    )
    yield {"type": "final", "result": outcome}


def start_run(request: str, *, chain, registry: RunRegistry) -> PausedRun:
    """Run the draft chain to completion, then PAUSE (stash + decline to execute)."""
    proposed = (chain.invoke({"request": request}) or "").strip()
    if not proposed:
        proposed = "(the model returned an empty draft)"
    run = PausedRun(run_id=uuid.uuid4().hex, request=request, proposed_action=proposed)
    registry.save(run)
    return run


def resume_run(
    run_id: str,
    *,
    approved: bool,
    registry: RunRegistry,
    feedback: str | None = None,
) -> dict:
    """Resume the paused run; pop it so it can't resume twice."""
    run = registry.pop(run_id)
    if run is None:
        raise UnknownRunError(run_id)

    if approved:
        run.status = "executed"
        result = f"{run.proposed_action}\n\n[ACTION EXECUTED / message sent]"
        return {"status": "executed", "result": result}

    run.status = "rejected"
    return {"status": "rejected", "result": None, "feedback": feedback}
