# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (raw-api). See raw-api/10-hitl-approval/README.md
"""Human-in-the-loop approval core (raw-api approach).

The flow is a two-phase workflow split by a human checkpoint:

1. ``draft_action`` — the LLM drafts a proposed high-risk action (e.g. a short
   refund-approval message) from the incoming request. The workflow then PAUSES.
2. ``execute_action`` / ``reject_action`` — once a human decides, we resume from
   the saved state and either mark the action sent (executed) or rejected.

Because the raw-api approach has **no native interrupt/checkpoint primitive**,
we hand-build the pause: we persist the paused run (proposed action + original
request) in a ``CheckpointStore`` keyed by ``run_id``. ``/resume`` looks the run
up, continues, and then locks/removes it so a run can only terminate once. See
the README "Why this is impractical in raw API" section — langgraph gets this
plumbing for free via ``interrupt()`` + a checkpointer.

The LLM call is injected (a ``Callable[[str], str]``), so unit tests run offline.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

LLMCall = Callable[[str], str]

DRAFT_SYSTEM_PROMPT = (
    "You are an operations assistant. A human will review and approve your work "
    "before it takes effect. Given a request for a high-risk action, draft a "
    "single short, professional message that performs the action (for example a "
    "refund-approval note to a customer). Output ONLY the message text, no "
    "preamble, no explanation, at most 3 sentences."
)


def draft_prompt(request: str) -> str:
    """Build the user prompt that asks the model to draft the proposed action."""
    return f"Request: {request}\n\nDraft the action message:"


# --------------------------------------------------------------------------- #
# Manual checkpoint store — the part langgraph gives you for free.
# --------------------------------------------------------------------------- #
@dataclass
class PausedRun:
    """A workflow paused at the human-approval checkpoint."""

    run_id: str
    request: str
    proposed_action: str
    status: str = "awaiting_approval"  # awaiting_approval | executed | rejected


@dataclass
class CheckpointStore:
    """In-memory, thread-safe persistence of paused runs keyed by run_id.

    A real system would use SQLite/Redis; the point is that *we* hand-roll the
    persistence the framework would otherwise own. Terminal resumes pop the run
    so it can never be resumed twice.
    """

    _runs: dict[str, PausedRun] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, run: PausedRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get(self, run_id: str) -> Optional[PausedRun]:
        with self._lock:
            return self._runs.get(run_id)

    def pop(self, run_id: str) -> Optional[PausedRun]:
        """Atomically fetch-and-remove a run (used on terminal resume)."""
        with self._lock:
            return self._runs.pop(run_id, None)


# --------------------------------------------------------------------------- #
# Phase 1 — draft then pause.
# --------------------------------------------------------------------------- #
def iter_start_run(
    request: str, *, llm_stream, store: CheckpointStore
) -> Iterator[dict]:
    """Draft with streaming, then park. See docs/streaming.md.

    Yields ``token`` events while the draft is written, then a single
    ``awaiting_approval`` event carrying the run id — **and stops there**.

    The stream deliberately ends at the gate. Holding a connection open across a
    human decision is exactly the fragility a durable approval workflow exists to
    avoid: the approver may take minutes or days, and any dropped socket, proxy
    timeout or server restart would lose the run. The checkpoint is the contract;
    the connection is not. `POST /resume/stream` opens a fresh stream once the
    decision arrives.
    """
    pieces: list[str] = []
    for piece in llm_stream(draft_prompt(request)):
        pieces.append(piece)
        yield {"type": "token", "text": piece}

    proposed = "".join(pieces).strip() or "(the model returned an empty draft)"
    run = PausedRun(
        run_id=uuid.uuid4().hex,
        request=request,
        proposed_action=proposed,
    )
    store.save(run)
    yield {
        "type": "awaiting_approval",
        "run_id": run.run_id,
        "proposed_action": run.proposed_action,
    }


def iter_resume_run(
    run_id: str,
    *,
    approved: bool,
    store: CheckpointStore,
    feedback: str | None = None,
) -> Iterator[dict]:
    """Resume a parked run as a (short) second stream.

    Emits the decision, then the outcome. No ``token`` events here: approving
    executes the already-drafted action rather than generating new text — which
    is itself the point of the gate. The human approved *that* text, so nothing
    may be regenerated after the fact.
    """
    yield {"type": "decision", "approved": approved}
    outcome = resume_run(run_id, approved=approved, store=store, feedback=feedback)
    yield {"type": "final", "result": outcome}


def start_run(request: str, *, llm_call: LLMCall, store: CheckpointStore) -> PausedRun:
    """Draft the proposed action and persist a paused run. Returns the PausedRun."""
    proposed = llm_call(draft_prompt(request)).strip()
    if not proposed:
        proposed = "(the model returned an empty draft)"
    run = PausedRun(
        run_id=uuid.uuid4().hex,
        request=request,
        proposed_action=proposed,
    )
    store.save(run)
    return run


# --------------------------------------------------------------------------- #
# Phase 2 — resume from saved state.
# --------------------------------------------------------------------------- #
class UnknownRunError(KeyError):
    """Raised when /resume is given a run_id that is not (or no longer) paused."""


def resume_run(
    run_id: str,
    *,
    approved: bool,
    store: CheckpointStore,
    feedback: str | None = None,
) -> dict:
    """Continue a paused run. Pops it from the store so it can't resume twice.

    approved=True  -> {"status": "executed", "result": "<action> [SENT]"}
    approved=False -> {"status": "rejected", "result": null, "feedback": ...}
    unknown run_id -> raises UnknownRunError (mapped to HTTP 404 in main.py).
    """
    run = store.pop(run_id)
    if run is None:
        raise UnknownRunError(run_id)

    if approved:
        run.status = "executed"
        result = f"{run.proposed_action}\n\n[ACTION EXECUTED / message sent]"
        return {"status": "executed", "result": result}

    run.status = "rejected"
    return {"status": "rejected", "result": None, "feedback": feedback}
