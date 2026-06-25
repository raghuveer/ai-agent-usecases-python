"""Hand-coded multi-agent orchestration (raw-api approach).

This is the raw-api point — and, for THIS use case, the raw-api *pain*. There
is no framework giving us sub-agent nodes, shared state, or routing. We wire the
three role-prompted sub-agents together by hand:

    orchestrator
      └─ researcher  : deterministic offline `research()` over data/corpus/*.md
      └─ writer      : LLM drafts a short summary from the research bullets
      └─ reviewer    : LLM returns a critique + an APPROVED: yes/no verdict
      └─ (revise loop): if rejected, writer redrafts once using the critique

Each sub-agent is a plain function. The orchestrator threads each one's output
to the next by hand, runs the bounded revise loop with plain Python, and packs
the aggregate response. ``llm_call(system, user) -> str`` is injected so unit
tests script every sub-agent's reply and never touch the network.

Text protocol, not provider-native function-calling: the reviewer emits a line
``APPROVED: yes`` / ``APPROVED: no`` that we parse with a redaction-tolerant
regex (the gateway's PII filter can mangle stray proper nouns, so we keep the
verdict token plain and the parser forgiving).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .search import format_research, research

# A sub-agent LLM call maps (system_prompt, user_message) -> reply text.
LLMCall = Callable[[str, str], str]


# --------------------------------------------------------------------------- #
# Role prompts (sub-agent personas)
# --------------------------------------------------------------------------- #
WRITER_SYSTEM = (
    "You are the writer agent in a small team. Using ONLY the research notes you "
    "are given, write a clear summary of the topic in 2 to 4 sentences. Do not "
    "invent facts that are not in the notes. Output only the summary text."
)

REVIEWER_SYSTEM = (
    "You are the reviewer agent in a small team. You receive research notes and a "
    "draft summary. Check that the draft is accurate against the notes, is clear, "
    "and invents nothing. Respond in EXACTLY this format:\n"
    "Critique: <one or two sentences>\n"
    "APPROVED: <yes or no>\n"
    "Approve (yes) when the draft is faithful to the notes and reads well. "
    "Reject (no) only when it contradicts the notes or is unclear."
)


def writer_user(topic: str, research_block: str, critique: str | None = None) -> str:
    """Build the writer's user message; include critique on a revise pass."""
    parts = [f"Topic: {topic}", "", "Research notes:", research_block]
    if critique:
        parts += [
            "",
            "A reviewer rejected your previous draft with this feedback. "
            "Revise to address it:",
            critique,
        ]
    parts += ["", "Write the summary now."]
    return "\n".join(parts)


def reviewer_user(topic: str, research_block: str, draft: str) -> str:
    return (
        f"Topic: {topic}\n\nResearch notes:\n{research_block}\n\n"
        f"Draft summary:\n{draft}\n\nReview it now."
    )


# --------------------------------------------------------------------------- #
# Reviewer verdict parsing (redaction-tolerant)
# --------------------------------------------------------------------------- #
_APPROVED_RE = re.compile(r"APPROVED\s*[:\-]?\s*(yes|no|true|false)", re.IGNORECASE)


def parse_approved(review_text: str) -> bool:
    """Parse the reviewer's APPROVED verdict. Defaults to True if absent.

    Defaulting to approval keeps the pipeline from looping forever when the
    model omits the token; the bounded revise loop is the real safety net.
    """
    m = _APPROVED_RE.search(review_text)
    if not m:
        return True
    return m.group(1).lower() in {"yes", "true"}


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #
@dataclass
class MultiAgentResult:
    draft: str
    review: str
    approved: bool
    contributions: dict = field(default_factory=dict)
    revisions: int = 0  # how many times the writer redrafted


# --------------------------------------------------------------------------- #
# Sub-agents
# --------------------------------------------------------------------------- #
def researcher(topic: str, top_k: int) -> tuple[list[str], str]:
    """Deterministic researcher: gather corpus facts and render a bullet block."""
    facts = research(topic, top_k=top_k)
    return facts, format_research(facts)


def writer(llm_call: LLMCall, topic: str, research_block: str,
           critique: str | None = None) -> str:
    return llm_call(WRITER_SYSTEM, writer_user(topic, research_block, critique))


def reviewer(llm_call: LLMCall, topic: str, research_block: str,
             draft: str) -> tuple[str, bool]:
    review = llm_call(REVIEWER_SYSTEM, reviewer_user(topic, research_block, draft))
    return review, parse_approved(review)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def orchestrate(
    topic: str,
    *,
    llm_call: LLMCall,
    research_top_k: int = 4,
    max_revisions: int = 1,
) -> MultiAgentResult:
    """Run researcher → writer → reviewer with one bounded revise loop.

    Order is enforced by hand: research first, then the first draft, then the
    review; on rejection the writer redrafts (up to ``max_revisions`` times)
    using the latest critique, and the reviewer judges the new draft.
    """
    facts, research_block = researcher(topic, research_top_k)

    draft = writer(llm_call, topic, research_block)
    review, approved = reviewer(llm_call, topic, research_block, draft)

    revisions = 0
    while not approved and revisions < max_revisions:
        revisions += 1
        draft = writer(llm_call, topic, research_block, critique=review)
        review, approved = reviewer(llm_call, topic, research_block, draft)

    return MultiAgentResult(
        draft=draft,
        review=review,
        approved=approved,
        contributions={
            "research": research_block,
            "writer": draft,
            "reviewer": review,
        },
        revisions=revisions,
    )
