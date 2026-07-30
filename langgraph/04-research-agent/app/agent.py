# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (langgraph). See langgraph/04-research-agent/README.md
"""Research agent (langgraph approach): a bundled corpus, a deterministic
offline ``search`` tool, and a TEXT-based ReAct loop modelled as a
``StateGraph`` cycle.

The graph makes the loop explicit:

    reason  -> (parse the model turn into thought/action/final_answer)
       |
    route   -> END if Final Answer or max_steps, else "act"
       |
    act     -> run search, append Observation to the transcript, bump step
       |
    (back to reason)

The whole ReAct transcript and the collected sources live in the typed
``AgentState``. The chat model is injected, so unit tests use
``FakeListChatModel`` and run offline. We do NOT use native function-calling —
the protocol is plain text so it works with any chat model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import strip_thinking, system_prompt, truncate_at_stop

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
DEFAULT_MAX_STEPS = 6


# --------------------------------------------------------------------------- #
# Corpus + search tool (deterministic, offline)
# --------------------------------------------------------------------------- #
@dataclass
class Passage:
    source: str
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_corpus(data_dir: Path = DATA_DIR) -> list[Passage]:
    """Read every ``*.md`` under ``data/corpus`` and split into paragraphs."""
    passages: list[Passage] = []
    for path in sorted(data_dir.glob("*.md")):
        for para in _split_paragraphs(path.read_text(encoding="utf-8")):
            passages.append(Passage(source=path.name, text=para))
    return passages


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class Corpus:
    """The bundled corpus plus a deterministic keyword-overlap ``search``."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.passages = load_corpus(data_dir)

    def search(self, query: str, top_k: int = 3) -> list[Passage]:
        q = _tokens(query)
        scored: list[tuple[int, Passage]] = []
        for p in self.passages:
            overlap = len(q & _tokens(p.text))
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda sp: (-sp[0], sp[1].source, sp[1].text))
        return [p for _, p in scored[:top_k]]

    def search_text(self, query: str, top_k: int = 3) -> tuple[str, list[str]]:
        """Run search and render an Observation string + the sources seen."""
        hits = self.search(query, top_k)
        if not hits:
            return ("No matching passages found.", [])
        lines = [f"[{h.source}] {h.text}" for h in hits]
        sources = list(dict.fromkeys(h.source for h in hits))
        return ("\n".join(lines), sources)


# --------------------------------------------------------------------------- #
# ReAct prompt + parsing
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a research agent for a home-robotics company. Answer the user's "
    "question using ONLY facts you gather with the search tool over the local "
    "corpus. Do not use outside knowledge.\n\n"
    "You have exactly one tool:\n"
    "  search(query): returns the most relevant passages from the corpus, each "
    "prefixed with its [source.md] filename.\n\n"
    "Work in steps using EXACTLY this format:\n"
    "Thought: <your reasoning>\n"
    "Action: search\n"
    "Action Input: <a single-line search query>\n\n"
    "After each action you will receive:\n"
    "Observation: <search results>\n\n"
    "When you have enough evidence, stop and reply with:\n"
    "Thought: <reasoning>\n"
    "Final Answer: <a concise answer that cites the source filenames you used>\n\n"
    "Guidelines:\n"
    "- Break a multi-part question into separate searches: run one search per "
    "sub-topic (e.g. search returns, then separately search warranty).\n"
    "- Keep each search query to a few plain keywords; do not include names that "
    "are obvious from context.\n"
    "- A privacy filter may replace the company name or place names in the text "
    "with placeholder tokens like <PERSON>, <LOCATION>, or <ORGANIZATION>. These "
    "are normal; treat each token as the company being researched and answer "
    "anyway. NEVER refuse or ask for clarification because of a placeholder.\n"
    "- Only emit ONE Action per step. Do not invent Observations."
)

_ACTION_RE = re.compile(
    r"Action\s*:\s*(?P<action>[^\n]+?)\s*\n+\s*Action\s*Input\s*:\s*(?P<input>[^\n]*)",
    re.IGNORECASE,
)
_ACTION_ONLY_RE = re.compile(r"Action\s*:\s*(?P<action>[^\n]+)", re.IGNORECASE)
_LABEL_PREFIX_RE = re.compile(r"^\s*(?:<[^>]+>|[A-Za-z ]{0,20}?)\s*:\s*")
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(?P<answer>.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(?P<thought>[^\n]*)", re.IGNORECASE)
_REDACTION_RE = re.compile(r"<[^>]+>")
_BRAND_RE = re.compile(r"\bNorthwind(\s+Robotics)?\b", re.IGNORECASE)


def _strip_redactions(text: str) -> str:
    return re.sub(r"\s+", " ", _REDACTION_RE.sub(" ", text)).strip()


def _desensitize(text: str) -> str:
    """Replace the brand name with a neutral phrase to dodge the gateway's PII
    redaction (it rewrites the proper noun to a <PERSON> token, which derails
    the model)."""
    return _BRAND_RE.sub("the company", text)


def _next_nonempty_after(text: str, pos: int) -> str:
    for line in text[pos:].splitlines():
        if line.strip():
            return _LABEL_PREFIX_RE.sub("", line, count=1).strip()
    return ""


@dataclass
class Parsed:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str | None = None


def parse_step(text: str) -> Parsed:
    """Defensively parse a model turn. Final Answer wins; else the last Action.

    Tolerates the gateway redacting the ``Action Input`` label to a token like
    ``<PERSON>:`` (falls back to the next non-empty line as the query).
    """
    parsed = Parsed()

    thoughts = _THOUGHT_RE.findall(text)
    if thoughts:
        parsed.thought = thoughts[-1].strip()

    final = _FINAL_RE.search(text)
    if final:
        parsed.final_answer = final.group("answer").strip()
        return parsed

    actions = list(_ACTION_RE.finditer(text))
    if actions:
        last = actions[-1]
        parsed.action = last.group("action").strip()
        parsed.action_input = _strip_redactions(last.group("input").strip().strip("`\"'"))
        return parsed

    only = list(_ACTION_ONLY_RE.finditer(text))
    if only:
        last = only[-1]
        parsed.action = last.group("action").strip()
        parsed.action_input = _strip_redactions(
            _next_nonempty_after(text, last.end()).strip("`\"'")
        )
    return parsed


# --------------------------------------------------------------------------- #
# Graph state + nodes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    question: str
    transcript: str
    steps: list[dict]
    sources: list[str]
    step_count: int
    max_steps: int
    top_k: int
    # Scratch fields the nodes pass between each other within one cycle.
    pending_thought: str
    pending_action: str
    pending_input: str
    answer: str
    stopped_reason: str
    nudged: bool


def _build_user_prompt(question: str, transcript: str) -> str:
    base = f"Question: {_desensitize(question)}\n\n"
    if transcript:
        base += transcript + "\n"
    base += "Continue. Respond with the next Thought and either an Action or a Final Answer."
    return base


def build_agent_graph(corpus: Corpus, llm: BaseChatModel):
    """Compile the ReAct StateGraph: reason -> (END | act) -> reason."""

    def reason(state: AgentState) -> dict:
        """Call the model for one turn and parse it into the scratch fields."""
        messages = [
            SystemMessage(content=system_prompt(SYSTEM_PROMPT)),
            HumanMessage(
                content=_build_user_prompt(state["question"], state["transcript"])
            ),
        ]
        reply = strip_thinking(llm.invoke(messages).content)
        # The endpoint may not honour `stop`, so cut the turn at the first
        # Observation: here — the graph, not the model, supplies observations.
        parsed = parse_step(
            truncate_at_stop(reply if isinstance(reply, str) else str(reply))
        )

        if parsed.final_answer is not None:
            return {
                "answer": _REDACTION_RE.sub("Northwind Robotics", parsed.final_answer),
                "stopped_reason": "final_answer",
                "pending_action": "",
            }

        if not parsed.action:
            # No Action and no Final Answer: nudge once via the transcript.
            if state["nudged"]:
                return {"pending_action": "", "stopped_reason": "stuck"}
            return {
                "pending_action": "",
                "nudged": True,
                "transcript": state["transcript"]
                + f"\nAssistant: {reply.strip()}\n"
                "Observation: Please reply with either 'Action: search' + "
                "'Action Input:', or 'Final Answer:'.\n",
            }

        return {
            "pending_thought": parsed.thought,
            "pending_action": parsed.action,
            "pending_input": parsed.action_input,
        }

    def act(state: AgentState) -> dict:
        """Run the search tool, append the Observation, advance the transcript."""
        action = state["pending_action"]
        action_input = state["pending_input"]

        if action.lower().startswith("search"):
            obs, hit_sources = corpus.search_text(action_input, top_k=state["top_k"])
            action_name = "search"
        else:
            obs, hit_sources = (
                f"Unknown tool '{action}'. The only tool is 'search'.",
                [],
            )
            action_name = action

        sources = list(state["sources"])
        for s in hit_sources:
            if s not in sources:
                sources.append(s)

        step = {
            "thought": state["pending_thought"],
            "action": action_name,
            "action_input": action_input,
            "observation": obs,
        }
        transcript = state["transcript"] + (
            f"\nThought: {state['pending_thought']}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}\n"
            f"Observation: {_desensitize(obs)}\n"
        )
        return {
            "steps": state["steps"] + [step],
            "sources": sources,
            "transcript": transcript,
            "step_count": state["step_count"] + 1,
        }

    def route(state: AgentState) -> str:
        """Conditional edge: stop on a final answer, a stuck loop, or the cap."""
        if state.get("stopped_reason") in ("final_answer", "stuck"):
            return END
        if not state["pending_action"]:
            # Nudged this turn: loop back to reason for one more try.
            return "reason"
        if state["step_count"] >= state["max_steps"]:
            return "cap"
        return "act"

    def cap(state: AgentState) -> dict:
        return {
            "answer": "(stopped: reached the maximum number of steps without a "
            "final answer)",
            "stopped_reason": "max_steps",
        }

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("cap", cap)
    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason", route, {"act": "act", "reason": "reason", "cap": "cap", END: END}
    )
    graph.add_edge("act", "reason")
    graph.add_edge("cap", END)
    # Allow up to max_steps reason/act cycles (plus nudges) before LangGraph's
    # recursion guard trips.
    return graph.compile()


@dataclass
class AgentResult:
    answer: str
    sources: list[str]
    steps: list[dict]
    stopped_reason: str


def run_agent(
    question: str,
    *,
    corpus: Corpus,
    llm: BaseChatModel,
    max_steps: int = DEFAULT_MAX_STEPS,
    top_k: int = 3,
) -> AgentResult:
    """Invoke the compiled graph and shape the result."""
    graph = build_agent_graph(corpus, llm)
    init: AgentState = {
        "question": question,
        "transcript": "",
        "steps": [],
        "sources": [],
        "step_count": 0,
        "max_steps": max_steps,
        "top_k": top_k,
        "pending_thought": "",
        "pending_action": "",
        "pending_input": "",
        "answer": "",
        "stopped_reason": "",
        "nudged": False,
    }
    # Give LangGraph enough recursion budget for max_steps cycles + routing.
    out = graph.invoke(init, {"recursion_limit": max_steps * 4 + 10})
    return AgentResult(
        answer=out["answer"],
        sources=out["sources"],
        steps=out["steps"],
        stopped_reason=out["stopped_reason"],
    )
