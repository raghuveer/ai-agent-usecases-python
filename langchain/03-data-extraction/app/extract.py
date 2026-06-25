# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langchain). See langchain/03-data-extraction/README.md
"""Structured extraction core (langchain approach).

A small LCEL chain turns invoice text into a JSON string:

    prompt | llm | StrOutputParser

We then parse the JSON out of the reply and validate it against a fixed Pydantic
``Invoice`` model. On any parse/validation failure we re-run the chain ONCE with
the validation error fed back in, then give up (the HTTP layer returns 422).

The LLM is injected (a ``BaseChatModel``), so unit tests pass ``FakeListChatModel``
and run with no network.
"""
from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from .llm import model_profile
from .settings import get_settings


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

SYSTEM_PROMPT = (
    "You are a precise invoice data-extraction engine. "
    "Extract the fields from the invoice text and return them as a SINGLE JSON "
    "object that matches the given JSON schema. "
    "Return ONLY the JSON object — no prose, no markdown, no code fences. "
    "Use numbers (not strings) for monetary amounts. "
    "If a field is missing, use an empty string for text or 0 for numbers.\n\n"
    "JSON schema:\n{schema}"
)

USER_PROMPT = (
    "Extract the invoice fields from the text below and return ONLY the JSON "
    "object.{feedback}\n\nInvoice text:\n{text}"
)


def _system_prompt(model: str) -> str:
    """Disable qwen3 thinking mode by prefixing /no_think."""
    return model_profile(model)["thinking_prefix"] + SYSTEM_PROMPT


class ExtractionError(Exception):
    """Raised when the model output still fails validation after one retry."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# JSON extraction + validation
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
# Chain + extract with validate + retry-once
# --------------------------------------------------------------------------- #
def build_chain(llm: BaseChatModel, model_name: str):
    """Assemble the LCEL extraction chain (inputs -> JSON string).

    This is the ``text`` mode chain: the model returns a JSON string that the
    caller parses/validates. The ``native`` mode (``with_structured_output``)
    builds its own chain inside ``extract_invoice``.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_prompt(model_name)), ("human", USER_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


def build_native_chain(llm: BaseChatModel, model_name: str):
    """Assemble the native structured-output chain (inputs -> validated Invoice).

    Uses langchain's ``with_structured_output`` so the provider returns a parsed,
    schema-validated ``Invoice`` directly — no manual JSON extraction. Requires a
    model/route that supports structured output (e.g. OpenAI JSON mode, many
    LiteLLM routes); not all local models do.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_prompt(model_name)), ("human", USER_PROMPT)]
    )
    return prompt | llm.with_structured_output(Invoice)


def extract_invoice(
    text: str,
    llm: BaseChatModel,
    model_name: str,
    structured_mode: str | None = None,
) -> Invoice:
    """Extract an ``Invoice`` from raw text, validating with a single retry.

    ``structured_mode`` selects the parsing path (defaults to
    ``settings.llm_structured_mode``):

    - ``text``  (default): prompt asks for JSON, parse it out of the text reply.
    - ``native``: use the provider's native structured output
      (``llm.with_structured_output(Invoice)``), which returns a validated object.

    Both honour the same validate / retry-once contract.
    """
    if structured_mode is None:
        structured_mode = get_settings().llm_structured_mode

    if structured_mode == "native":
        return _extract_invoice_native(text, llm, model_name)
    return _extract_invoice_text(text, llm, model_name)


def _extract_invoice_text(text: str, llm: BaseChatModel, model_name: str) -> Invoice:
    """``text`` mode: parse JSON out of the model's text reply, retry once."""
    chain = build_chain(llm, model_name)

    reply = chain.invoke(
        {"schema": INVOICE_SCHEMA_JSON, "text": text, "feedback": ""}
    )
    try:
        return parse_and_validate(reply)
    except (ValueError, json.JSONDecodeError, ValidationError) as first_err:
        feedback = (
            "\nYour previous response could not be validated. "
            f"Error: {first_err}. "
            "Return ONLY a corrected JSON object matching the schema."
        )
        reply = chain.invoke(
            {"schema": INVOICE_SCHEMA_JSON, "text": text, "feedback": feedback}
        )
        try:
            return parse_and_validate(reply)
        except (ValueError, json.JSONDecodeError, ValidationError) as second_err:
            raise ExtractionError(
                f"extraction failed after retry: {second_err}", raw=reply
            ) from second_err


def _extract_invoice_native(text: str, llm: BaseChatModel, model_name: str) -> Invoice:
    """``native`` mode: provider returns a validated ``Invoice``, retry once."""
    chain = build_native_chain(llm, model_name)

    try:
        return chain.invoke(
            {"schema": INVOICE_SCHEMA_JSON, "text": text, "feedback": ""}
        )
    except (ValueError, ValidationError) as first_err:
        feedback = (
            "\nYour previous response could not be validated. "
            f"Error: {first_err}. "
            "Return ONLY a corrected JSON object matching the schema."
        )
        try:
            return chain.invoke(
                {"schema": INVOICE_SCHEMA_JSON, "text": text, "feedback": feedback}
            )
        except (ValueError, ValidationError) as second_err:
            raise ExtractionError(
                f"extraction failed after retry: {second_err}", raw=""
            ) from second_err
