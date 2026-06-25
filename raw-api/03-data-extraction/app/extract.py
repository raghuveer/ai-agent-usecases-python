# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (raw-api). See raw-api/03-data-extraction/README.md
"""Hand-written structured data extraction: prompt -> JSON -> validate -> retry.

This is the raw-api point: nothing is hidden behind a framework. We give the
model the target JSON schema, ask for JSON only, pull the JSON object out of the
reply by hand, validate it against a fixed Pydantic model, and — if validation
fails — call the model ONCE more with the validation error appended.

The LLM call is injectable so unit tests can swap in a canned-reply function and
never touch the network.
"""
from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel, ValidationError


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


# JSON schema handed to the model so it knows exactly what shape to return.
INVOICE_SCHEMA_JSON = json.dumps(Invoice.model_json_schema(), indent=2)

SYSTEM_PROMPT = (
    "You are a precise invoice data-extraction engine. "
    "Extract the fields from the invoice text the user provides and return them "
    "as a SINGLE JSON object that matches the given JSON schema. "
    "Return ONLY the JSON object — no prose, no markdown, no code fences. "
    "Use numbers (not strings) for monetary amounts. "
    "If a field is missing, use an empty string for text or 0 for numbers.\n\n"
    f"JSON schema:\n{INVOICE_SCHEMA_JSON}"
)


# An LLM call is `(system_prompt, user_prompt) -> reply`. Injectable for tests.
LLMCall = Callable[[str, str], str]


class ExtractionError(Exception):
    """Raised when the model output still fails validation after one retry."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# JSON extraction helpers
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> str:
    """Pull the first balanced ``{...}`` JSON object out of a model reply.

    Tolerates the model wrapping the JSON in markdown fences or surrounding
    prose. Returns the raw JSON substring (not yet parsed).
    """
    s = text.strip()
    # Strip a leading ```json / ``` fence if present.
    if s.startswith("```"):
        s = s.split("```", 2)
        s = s[1] if len(s) > 1 else text
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
    data = json.loads(raw)  # may raise json.JSONDecodeError
    return Invoice.model_validate(data)  # may raise ValidationError


# --------------------------------------------------------------------------- #
# Extraction with validate + retry-once
# --------------------------------------------------------------------------- #
def build_user_prompt(text: str) -> str:
    return (
        "Extract the invoice fields from the text below and return ONLY the JSON "
        "object.\n\nInvoice text:\n" + text
    )


def extract_invoice(text: str, *, llm_call: LLMCall) -> Invoice:
    """Extract an ``Invoice`` from raw text, validating with a single retry.

    1. Ask the model for JSON, parse + validate.
    2. On parse/validation failure, append the error to the prompt and ask once
       more.
    3. If it still fails, raise ``ExtractionError`` (the HTTP layer maps to 422).
    """
    user_prompt = build_user_prompt(text)
    reply = llm_call(SYSTEM_PROMPT, user_prompt)

    try:
        return parse_and_validate(reply)
    except (ValueError, json.JSONDecodeError, ValidationError) as first_err:
        # Retry ONCE, telling the model exactly what was wrong.
        retry_prompt = (
            user_prompt
            + "\n\nYour previous response could not be validated.\n"
            + f"Error: {first_err}\n"
            + "Return ONLY a corrected JSON object matching the schema."
        )
        reply = llm_call(SYSTEM_PROMPT, retry_prompt)
        try:
            return parse_and_validate(reply)
        except (ValueError, json.JSONDecodeError, ValidationError) as second_err:
            raise ExtractionError(
                f"extraction failed after retry: {second_err}", raw=reply
            ) from second_err
