# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (raw-api). See raw-api/04-research-agent/README.md
"""Research agent core: a bundled corpus, a deterministic offline ``search``
tool, and a hand-written TEXT-based ReAct loop.

This is the raw-api point: nothing is hidden behind a framework. We

- load the bundled ``data/corpus/*.md`` notes once,
- expose ONE tool, ``search(query)``, that scores corpus paragraphs by keyword
  overlap and returns the top snippets with their source filenames
  (deterministic, no network), and
- drive a text ReAct loop by hand: prompt the model for
  ``Thought / Action / Action Input`` (or ``Final Answer``), parse the last
  action with a regex, run the tool, append ``Observation:`` to the transcript,
  and loop until ``Final Answer`` or ``max_steps``.

We do NOT use provider-native function-calling; the protocol is plain text so it
works with any chat model. The LLM call is injected so unit tests run offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"

# A single LLM turn: takes the full transcript (one user message) and returns the
# assistant's text completion. Injectable so tests never touch the network.
LLMCall = Callable[[str], str]

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
        """Return the top_k passages by keyword overlap with ``query``.

        Scoring is simple and deterministic: |query_tokens ∩ passage_tokens|,
        with a tiny tie-break on source name + text so results are stable.
        """
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
        sources = list(dict.fromkeys(h.source for h in hits))  # dedupe, ordered
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

# Strict: the documented "Action: <tool>" + "Action Input: <query>" pair.
_ACTION_RE = re.compile(
    r"Action\s*:\s*(?P<action>[^\n]+?)\s*\n+\s*Action\s*Input\s*:\s*(?P<input>[^\n]*)",
    re.IGNORECASE,
)
# Fallback: just the "Action: <tool>" line, when the Action-Input LABEL got
# mangled (the gateway's PII filter rewrites the literal text "Action Input" to a
# redaction token like "<PERSON>:"). We then read the next non-empty line as the
# query, stripping any leading "Label:" prefix.
_ACTION_ONLY_RE = re.compile(r"Action\s*:\s*(?P<action>[^\n]+)", re.IGNORECASE)
_LABEL_PREFIX_RE = re.compile(r"^\s*(?:<[^>]+>|[A-Za-z ]{0,20}?)\s*:\s*")
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(?P<answer>.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(?P<thought>[^\n]*)", re.IGNORECASE)


_REDACTION_RE = re.compile(r"<[^>]+>")


def _strip_redactions(text: str) -> str:
    """Remove gateway PII-redaction tokens (e.g. ``<PERSON>``) from a string and
    collapse whitespace. The gateway rewrites some proper nouns/labels to such
    tokens; they are noise for our keyword search."""
    return re.sub(r"\s+", " ", _REDACTION_RE.sub(" ", text)).strip()


def _next_nonempty_after(text: str, pos: int) -> str:
    """Return the next non-empty line after byte offset ``pos`` in ``text``,
    with any leading ``Label:`` (incl. a ``<REDACTED>:`` token) stripped."""
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
    """Defensively parse a model turn into thought/action/final_answer.

    Final Answer wins if present. Otherwise we take the LAST Action/Action Input
    pair. Tolerates missing fields (returns empty strings).
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

    # Fallback: an "Action:" line whose "Action Input" label was redacted.
    only = list(_ACTION_ONLY_RE.finditer(text))
    if only:
        last = only[-1]
        parsed.action = last.group("action").strip()
        parsed.action_input = _strip_redactions(
            _next_nonempty_after(text, last.end()).strip("`\"'")
        )
    return parsed


@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_steps"


# --------------------------------------------------------------------------- #
# The ReAct loop
# --------------------------------------------------------------------------- #
# The gateway's PII filter redacts the proper noun "Northwind" to a <PERSON>
# token, which derails the model. We swap it for a neutral phrase in the text we
# SEND to the model (the corpus, the user's original question, and our response
# all keep the real name). Case-insensitive, word-boundary safe.
_BRAND_RE = re.compile(r"\bNorthwind(\s+Robotics)?\b", re.IGNORECASE)


def _desensitize(text: str) -> str:
    """Replace the brand name with a neutral phrase to dodge PII redaction."""
    return _BRAND_RE.sub("the company", text)


def _build_user_prompt(question: str, transcript: str) -> str:
    base = f"Question: {_desensitize(question)}\n\n"
    if transcript:
        base += transcript + "\n"
    base += "Continue. Respond with the next Thought and either an Action or a Final Answer."
    return base


def run_agent(
    question: str,
    *,
    corpus: Corpus,
    llm_call: LLMCall,
    max_steps: int = DEFAULT_MAX_STEPS,
    top_k: int = 3,
) -> AgentResult:
    """Drive the text ReAct loop until Final Answer or max_steps.

    ``llm_call`` receives the running user prompt (question + transcript) and
    returns the assistant's text. We parse it, run ``search`` on any action,
    append the Observation, and loop.
    """
    transcript = ""
    steps: list[Step] = []
    sources: list[str] = []
    nudged = False

    for _ in range(max_steps):
        reply = llm_call(_build_user_prompt(question, transcript))
        parsed = parse_step(reply)

        if parsed.final_answer is not None:
            # Restore the company name where the gateway's PII filter blanked it.
            answer = _REDACTION_RE.sub("Northwind Robotics", parsed.final_answer)
            return AgentResult(
                answer=answer,
                sources=sources,
                steps=steps,
                stopped_reason="final_answer",
            )

        if not parsed.action:
            # No Action and no Final Answer: nudge once, then stop.
            if nudged:
                break
            nudged = True
            transcript += (
                f"\nAssistant: {reply.strip()}\n"
                "Observation: Please reply with either 'Action: search' + "
                "'Action Input:', or 'Final Answer:'.\n"
            )
            continue

        # Run the tool. The only tool is `search`; anything else is reported back.
        if parsed.action.lower().startswith("search"):
            obs, hit_sources = corpus.search_text(parsed.action_input, top_k=top_k)
            for s in hit_sources:
                if s not in sources:
                    sources.append(s)
        else:
            obs = f"Unknown tool '{parsed.action}'. The only tool is 'search'."

        steps.append(
            Step(
                thought=parsed.thought,
                action="search" if parsed.action.lower().startswith("search") else parsed.action,
                action_input=parsed.action_input,
                observation=obs,
            )
        )
        # Desensitize the brand name in what we feed BACK to the model (the
        # `steps`/response keep the real observation text).
        transcript += (
            f"\nThought: {parsed.thought}\n"
            f"Action: {parsed.action}\n"
            f"Action Input: {parsed.action_input}\n"
            f"Observation: {_desensitize(obs)}\n"
        )

    # Hit the step cap without a Final Answer.
    return AgentResult(
        answer="(stopped: reached the maximum number of steps without a final answer)",
        sources=sources,
        steps=steps,
        stopped_reason="max_steps",
    )
