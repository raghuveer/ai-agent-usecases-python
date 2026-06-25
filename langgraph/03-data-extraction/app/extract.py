# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langgraph). See langgraph/03-data-extraction/README.md
"""Structured extraction core for UC3 as a tiny langgraph StateGraph.

The graph makes the validate-and-retry loop explicit as edges:

    extract -> validate -> (retry? back to extract : END)

- ``extract`` calls the LLM (injected) with the schema + any prior validation
  error, and stores the raw reply.
- ``validate`` parses the first balanced JSON object out of the reply and
  validates it against a fixed Pydantic ``Invoice``. On success it stores the
  invoice; on failure it stores the error.
- A conditional edge loops back to ``extract`` exactly ONCE (retry), then ends.

This is the langgraph point: the control flow (retry-on-invalid) lives in the
graph topology, not in an imperative try/except. The LLM is injected, so unit
tests pass a ``FakeListChatModel`` and run fully offline.
"""
from __future__ import annotations

import json
from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from .llm import system_prompt
from .settings import Settings, get_settings

# Cap the number of LLM attempts: 1 initial + 1 retry.
MAX_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Fixed target schema
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    description: str
    amount: float


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    date: str
    total: float
    line_items: list[LineItem]


INVOICE_SCHEMA_JSON = json.dumps(Invoice.model_json_schema(), indent=2)

SYSTEM_INSTRUCTION = (
    "You are a precise invoice data-extraction engine. "
    "Extract the fields from the invoice text and return them as a SINGLE JSON "
    "object that matches the given JSON schema. "
    "Return ONLY the JSON object — no prose, no markdown, no code fences. "
    "Use numbers (not strings) for monetary amounts. "
    "If a field is missing, use an empty string for text or 0 for numbers.\n\n"
    f"JSON schema:\n{INVOICE_SCHEMA_JSON}"
)


class ExtractionError(Exception):
    """Raised when the model output still fails validation after one retry."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# JSON extraction + validation helpers
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> str:
    """Pull the first balanced ``{...}`` JSON object out of a model reply."""
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else text
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ValueError("unbalanced JSON braces in model output")


def parse_and_validate(text: str) -> Invoice:
    """Parse the model reply into a validated ``Invoice`` (raises on failure)."""
    raw = extract_json_object(text)
    data = json.loads(raw)
    return Invoice.model_validate(data)


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class ExtractState(TypedDict):
    text: str
    attempts: int
    raw: str
    error: str
    invoice: Optional[Invoice]


def build_extract_graph(llm: BaseChatModel, settings: Settings | None = None):
    """Compile the extract -> validate -> (retry|END) graph. LLM is injected."""
    settings = settings or get_settings()

    def extract(state: ExtractState) -> dict:
        feedback = ""
        if state.get("error"):
            feedback = (
                "\n\nYour previous response could not be validated.\n"
                f"Error: {state['error']}\n"
                "Return ONLY a corrected JSON object matching the schema."
            )
        messages = [
            SystemMessage(content=system_prompt(SYSTEM_INSTRUCTION, settings)),
            HumanMessage(
                content=(
                    "Extract the invoice fields from the text below and return "
                    "ONLY the JSON object." + feedback + "\n\nInvoice text:\n"
                    + state["text"]
                )
            ),
        ]
        result = llm.invoke(messages)
        return {"raw": result.content, "attempts": state.get("attempts", 0) + 1}

    def validate(state: ExtractState) -> dict:
        try:
            invoice = parse_and_validate(state["raw"])
            return {"invoice": invoice, "error": ""}
        except (ValueError, json.JSONDecodeError, ValidationError) as err:
            return {"invoice": None, "error": str(err)}

    def route(state: ExtractState) -> str:
        # Valid -> done. Invalid but attempts left -> retry. Otherwise -> done.
        if state.get("invoice") is not None:
            return "done"
        if state["attempts"] < MAX_ATTEMPTS:
            return "retry"
        return "done"

    graph = StateGraph(ExtractState)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate", route, {"retry": "extract", "done": END}
    )
    return graph.compile()


def extract_invoice(text: str, llm: BaseChatModel,
                    settings: Settings | None = None) -> Invoice:
    """Run the extraction graph and return a validated ``Invoice``.

    Raises ``ExtractionError`` if the model output is still invalid after the
    single retry (the HTTP layer maps that to 422).
    """
    graph = build_extract_graph(llm, settings)
    out = graph.invoke(
        {"text": text, "attempts": 0, "raw": "", "error": "", "invoice": None}
    )
    if out.get("invoice") is None:
        raise ExtractionError(
            f"extraction failed after retry: {out.get('error')}",
            raw=out.get("raw", ""),
        )
    return out["invoice"]
