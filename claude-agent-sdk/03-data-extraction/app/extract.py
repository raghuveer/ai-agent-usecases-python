# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC03 Data extraction (claude-agent-sdk). See claude-agent-sdk/03-data-extraction/README.md
"""Structured extraction via *tool-as-schema*.

The other three approaches ask for JSON in the prompt, then parse and repair the
reply. The Agent SDK's idiom is different: declare a tool whose `input_schema`
*is* the target schema, and tell the agent its job is to call that tool. The
structured record then arrives as the tool's **input** — already a dict, already
shaped by the tool-calling machinery, never a string that needs parsing out of
prose.

    text ──► agent ──► emit_invoice(invoice_number=..., total=...) ──► dict

There is no JSON to extract from a reply, no markdown fences to strip, and no
"the model prefixed 'Here is the JSON:'" failure mode. What is *not* guaranteed
is semantic correctness — the agent can still call the tool with a wrong total —
so the tool input is validated against a Pydantic model and any failure is
reported rather than swallowed.

**Fit note.** This is an agent harness doing a one-shot job: no loop is needed,
and the SDK's built-in tools go unused. It works and it is clean, but the harness
earns less here than in the agentic use cases. See the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import BaseModel, Field, ValidationError

from .agent import Runner, build_options, default_runner, outcome_of
from .settings import Settings


class LineItem(BaseModel):
    description: str
    amount: float


class Invoice(BaseModel):
    """The extraction target. This is the contract, in one place."""

    invoice_number: str = Field(max_length=100)
    vendor: str = Field(max_length=200)
    invoice_date: str = Field(max_length=40, description="ISO date if determinable")
    currency: str = Field(max_length=10)
    total: float
    line_items: list[LineItem] = Field(default_factory=list)


# JSON Schema mirroring Invoice. Written explicitly (rather than generated from
# the model) because it is the prompt-facing contract: the field descriptions
# below are instructions the agent reads.
INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "description": "Invoice/reference number."},
        "vendor": {"type": "string", "description": "Company that issued the invoice."},
        "invoice_date": {"type": "string", "description": "Invoice date, ISO 8601 if determinable."},
        "currency": {"type": "string", "description": "ISO currency code, e.g. USD, EUR, INR."},
        "total": {"type": "number", "description": "Grand total as a number, no symbols or separators."},
        "line_items": {
            "type": "array",
            "description": "Each billed line.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["description", "amount"],
            },
        },
    },
    "required": ["invoice_number", "vendor", "invoice_date", "currency", "total"],
}

EMIT_TOOL = "mcp__extract__emit_invoice"


@tool(
    "emit_invoice",
    "Emit the extracted invoice as structured fields. Call this exactly once.",
    INVOICE_SCHEMA,
)
async def emit_invoice(args: dict[str, Any]) -> dict[str, Any]:
    """Sink for the structured record — the value is the *input*, not the output."""
    return {"content": [{"type": "text", "text": "Recorded."}]}


SYSTEM_PROMPT = """You extract invoice data from raw text.

Call the emit_invoice tool exactly once with the fields you extracted. That call
is your entire job — do not summarise or explain in prose.

Rules:
- Copy values from the document; never invent one.
- `total` is a number: no currency symbols, no thousands separators.
- If a required field is genuinely absent, use the empty string (or 0 for total)
  rather than guessing."""


def build_extract_server():
    return create_sdk_mcp_server(name="extract", version="1.0.0", tools=[emit_invoice])


@dataclass
class ExtractResult:
    valid: bool
    invoice: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    num_turns: int = 0
    cost_usd: float = 0.0
    stop_reason: str = "end_turn"


def _validate(payload: dict[str, Any]) -> ExtractResult:
    try:
        invoice = Invoice.model_validate(payload)
    except ValidationError as exc:
        return ExtractResult(
            valid=False,
            errors=[
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ],
        )
    return ExtractResult(valid=True, invoice=invoice.model_dump())


async def extract(
    document: str, settings: Settings, runner: Runner | None = None
) -> ExtractResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[EMIT_TOOL],
        mcp_servers={"extract": build_extract_server()},
        # One tool call is the whole task; no room needed to wander.
        max_turns=3,
    )
    result = await runner(document, options)

    calls = [c for c in result.tool_calls if c.name == EMIT_TOOL]
    if not calls:
        return ExtractResult(
            valid=False,
            errors=["agent did not call emit_invoice"],
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            stop_reason=outcome_of(result),
        )

    # If the agent called it more than once, the last call is its final answer.
    out = _validate(calls[-1].input)
    out.num_turns = result.num_turns
    out.cost_usd = result.cost_usd
    out.stop_reason = outcome_of(result)
    return out
