# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langgraph). See langgraph/07-multi-agent/README.md
"""UC7 multi-agent as a langgraph StateGraph — the showcase approach.

This is LangGraph's native territory: an orchestrator routing work between
specialised sub-agents over one shared, typed state. Each sub-agent is a **node**;
the reviewer's reject → writer revise loop is a **conditional edge**; everything
flows through a single ``MultiAgentState``:

    researcher ──► writer ──► reviewer ──► (route) ──approved──► END
                     ▲                        │
                     └──────reject (revise)───┘   (capped at max_revisions)

- **researcher**: deterministic, offline ``research()`` over data/corpus/*.md.
  Fills ``research`` in the shared state — the writer reads it from there.
- **writer**: role-prompted LLM node. Drafts from ``research`` (and, on a revise
  pass, from the reviewer's ``review`` critique). Writes ``draft``.
- **reviewer**: role-prompted LLM node. Reads ``draft``, writes ``review`` and
  the ``approved`` flag.
- **route** (conditional edge): if approved, or the revision cap is hit, go to
  END; otherwise bump ``revisions`` and loop back to the writer.

The shared state is the point: no sub-agent threads data to the next by hand —
each node reads what it needs from the state and writes its slice back, and the
orchestrator's routing is one declarative conditional edge. Contrast the
hand-wired ``raw-api/07-multi-agent``. The LLM is injected, so unit tests pass a
``FakeListChatModel`` and the whole graph runs offline.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import ThinkFilter, strip_thinking, system_prefix
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


# --------------------------------------------------------------------------- #
# Shared typed state — every sub-agent reads/writes its slice here.
# --------------------------------------------------------------------------- #
class MultiAgentState(TypedDict):
    topic: str
    research_top_k: int
    max_revisions: int
    research: str            # researcher → (writer, reviewer)
    draft: str               # writer → reviewer
    review: str              # reviewer → (writer on revise, response)
    approved: bool           # reviewer verdict
    revisions: int           # how many revise cycles have run


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


def _text(reply) -> str:
    return strip_thinking(
        reply.content if isinstance(reply, AIMessage) else str(reply)
    )


# --------------------------------------------------------------------------- #
# Graph construction — orchestrator over sub-agent nodes.
# --------------------------------------------------------------------------- #
def build_multi_agent_graph(llm: BaseChatModel):
    """Compile researcher → writer → reviewer → (revise loop | END)."""

    def researcher(state: MultiAgentState) -> dict:
        """Deterministic sub-agent: gather corpus facts into shared state."""
        facts = research(state["topic"], top_k=state["research_top_k"])
        return {"research": format_research(facts)}

    def writer(state: MultiAgentState) -> dict:
        """LLM sub-agent: draft from the research (+ critique on a revise pass)."""
        user = [f"Topic: {state['topic']}", "", "Research notes:", state["research"]]
        if state.get("review") and not state.get("approved", False):
            user += [
                "",
                "A reviewer rejected your previous draft with this feedback. "
                "Revise to address it:",
                state["review"],
            ]
        user += ["", "Write the summary now."]
        reply = llm.invoke([
            SystemMessage(content=system_prefix(WRITER_SYSTEM)),
            HumanMessage(content="\n".join(user)),
        ])
        return {"draft": _text(reply).strip()}

    def reviewer(state: MultiAgentState) -> dict:
        """LLM sub-agent: critique the draft and set the approval flag."""
        user = (
            f"Topic: {state['topic']}\n\nResearch notes:\n{state['research']}\n\n"
            f"Draft summary:\n{state['draft']}\n\nReview it now."
        )
        reply = llm.invoke([
            SystemMessage(content=system_prefix(REVIEWER_SYSTEM)),
            HumanMessage(content=user),
        ])
        review = _text(reply).strip()
        return {"review": review, "approved": parse_approved(review)}

    def route(state: MultiAgentState) -> str:
        """Conditional edge: END on approval / cap, else revise (loop to writer)."""
        if state.get("approved", False):
            return "end"
        if state["revisions"] >= state["max_revisions"]:
            return "end"
        return "revise"

    def revise(state: MultiAgentState) -> dict:
        """Bump the revision counter before looping back to the writer."""
        return {"revisions": state["revisions"] + 1}

    graph = StateGraph(MultiAgentState)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("revise", revise)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    # reviewer decides: stop (END) or loop back through revise → writer.
    graph.add_conditional_edges("reviewer", route, {"revise": "revise", "end": END})
    graph.add_edge("revise", "writer")
    return graph.compile()


def _initial_state(
    topic: str, research_top_k: int, max_revisions: int
) -> MultiAgentState:
    """Starting state — shared by the blocking and streaming paths."""
    return {
        "topic": topic,
        "research_top_k": research_top_k,
        "max_revisions": max_revisions,
        "research": "",
        "draft": "",
        "review": "",
        "approved": False,
        "revisions": 0,
    }


def iter_multi_agent(
    topic: str,
    *,
    llm: BaseChatModel,
    research_top_k: int = 4,
    max_revisions: int = 1,
) -> Iterator[dict]:
    """The team as a stream of events. See docs/streaming.md.

    Like `langgraph/08`, this streams token chunks *and* node transitions
    (``stream_mode=["messages", "updates"]``). For a multi-agent team the node
    frames are the hand-offs themselves — researcher, writer, reviewer, revise —
    reported by the framework rather than narrated by our own code, which is the
    difference from `raw-api/07`.
    """
    graph = build_multi_agent_graph(llm)
    init = _initial_state(topic, research_top_k, max_revisions)

    think = ThinkFilter()
    state: dict = {}

    for mode, chunk in graph.stream(
        init,
        config={"recursion_limit": max_revisions * 3 + 6},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            message, _meta = chunk
            raw = getattr(message, "content", "") or ""
            visible = think.feed(raw if isinstance(raw, str) else str(raw))
            if visible:
                yield {"type": "token", "text": visible}
            continue

        for node, delta in (chunk or {}).items():
            # The graph reports the hand-off; we do not have to track it.
            yield {"type": "role", "name": node}
            state.update(delta or {})
            for artifact in ("research", "draft", "review"):
                if delta and artifact in delta and delta[artifact]:
                    yield {
                        "type": "artifact",
                        "name": artifact,
                        "text": delta[artifact],
                    }

    tail = think.flush()
    if tail:
        yield {"type": "token", "text": tail}

    yield {"type": "final", "result": _normalise(state)}


def _normalise(out: dict) -> dict:
    """Graph state -> the response contract."""
    return {
        "draft": out.get("draft", ""),
        "review": out.get("review", ""),
        "approved": out.get("approved", False),
        "revisions": out.get("revisions", 0),
        "contributions": {
            "research": out.get("research", ""),
            "writer": out.get("draft", ""),
            "reviewer": out.get("review", ""),
        },
    }


def run_multi_agent(
    topic: str,
    *,
    llm: BaseChatModel,
    research_top_k: int = 4,
    max_revisions: int = 1,
    callbacks: list | None = None,
) -> dict:
    """Run the compiled graph and return a normalised result dict.

    ``callbacks`` go in the graph's run config, not on the model: that is the
    only place LangGraph reports which *node* is executing, so a handler
    attached to the LLM records every call and none of the route.

    Returns ``{"draft", "review", "approved", "contributions", "revisions"}``.
    """
    graph = build_multi_agent_graph(llm)
    init = _initial_state(topic, research_top_k, max_revisions)
    # Each revise cycle adds writer+reviewer+revise nodes; give recursion headroom.
    config: dict = {"recursion_limit": max_revisions * 3 + 6}
    if callbacks:
        config["callbacks"] = callbacks
    out = graph.invoke(init, config=config)
    return {
        "draft": out["draft"],
        "review": out["review"],
        "approved": out["approved"],
        "revisions": out["revisions"],
        "contributions": {
            "research": out["research"],
            "writer": out["draft"],
            "reviewer": out["review"],
        },
    }
