"""Multi-agent pipeline as LangChain role chains (langchain approach).

Each sub-agent is a small **LCEL chain**: a ``ChatPromptTemplate`` piped into the
injected chat model piped into a ``StrOutputParser``. The orchestrator composes
them sequentially and wraps a plain-Python revise loop around the writer and
reviewer:

    researcher (deterministic search)  →  writer chain  →  reviewer chain
                                              ▲                  │ reject
                                              └──────────────────┘  (revise, capped)

LangChain gives us composable chains, but it has no native sub-agent/shared-state
primitive — so the routing (reject → revise, the cap, the aggregation) is still
plain Python here. The ``langgraph`` sibling expresses that routing as a graph.

The chat model is injected, so unit tests pass a ``FakeListChatModel`` and the
whole pipeline runs offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import system_prefix
from .search import format_research, research

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


def build_writer_chain(llm: BaseChatModel) -> Runnable:
    """LCEL chain for the writer sub-agent: prompt | llm | str parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prefix(WRITER_SYSTEM)),
        ("human",
         "Topic: {topic}\n\nResearch notes:\n{research}\n{critique}\n"
         "Write the summary now."),
    ])
    return prompt | llm | StrOutputParser()


def build_reviewer_chain(llm: BaseChatModel) -> Runnable:
    """LCEL chain for the reviewer sub-agent: prompt | llm | str parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prefix(REVIEWER_SYSTEM)),
        ("human",
         "Topic: {topic}\n\nResearch notes:\n{research}\n\n"
         "Draft summary:\n{draft}\n\nReview it now."),
    ])
    return prompt | llm | StrOutputParser()


def _critique_block(critique: str | None) -> str:
    if not critique:
        return ""
    return (
        "\nA reviewer rejected your previous draft with this feedback. "
        f"Revise to address it:\n{critique}\n"
    )


# --------------------------------------------------------------------------- #
# Reviewer verdict parsing (redaction-tolerant)
# --------------------------------------------------------------------------- #
_APPROVED_RE = re.compile(r"APPROVED\s*[:\-]?\s*(yes|no|true|false)", re.IGNORECASE)


def parse_approved(review_text: str) -> bool:
    """Parse the reviewer's APPROVED verdict. Defaults to True if absent."""
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
    revisions: int = 0


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def orchestrate(
    topic: str,
    *,
    llm: BaseChatModel,
    research_top_k: int = 4,
    max_revisions: int = 1,
) -> MultiAgentResult:
    """Run researcher → writer → reviewer with one bounded revise loop."""
    research_block = format_research(research(topic, top_k=research_top_k))

    writer = build_writer_chain(llm)
    reviewer = build_reviewer_chain(llm)

    draft = writer.invoke(
        {"topic": topic, "research": research_block, "critique": ""}
    ).strip()
    review = reviewer.invoke(
        {"topic": topic, "research": research_block, "draft": draft}
    ).strip()
    approved = parse_approved(review)

    revisions = 0
    while not approved and revisions < max_revisions:
        revisions += 1
        draft = writer.invoke({
            "topic": topic,
            "research": research_block,
            "critique": _critique_block(review),
        }).strip()
        review = reviewer.invoke(
            {"topic": topic, "research": research_block, "draft": draft}
        ).strip()
        approved = parse_approved(review)

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
