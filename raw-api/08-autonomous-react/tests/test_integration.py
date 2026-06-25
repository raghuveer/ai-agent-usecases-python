"""Integration test — drives the real ReAct loop via the gateway.

Two-tool chain: search Northwind's return window (30) then calculator doubles
it (60). Gated: runs only when RUN_INTEGRATION=1.

This use case defaults to ``claude-haiku-4-5`` (see README "Model note"): the
free local Qwen could not reliably drive the two-tool text-ReAct chain, so the
test is also marked ``anthropic``.
"""
from __future__ import annotations

import os

import pytest

from app import llm, react
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_two_tool_chain_against_gateway():
    settings = get_settings()
    client = llm.build_client(settings)

    def llm_call(messages: list[dict]) -> str:
        return llm.chat(
            client, model=settings.llm_model, messages=messages, stop=["Observation:"]
        )

    result = react.run_react(
        "Find Northwind's return window in days, then tell me what double that is.",
        llm_call=llm_call,
        max_steps=settings.max_steps,
    )

    assert result.answer.strip()
    assert "60" in result.answer
    # The model should have used both tools somewhere along the way.
    actions = {s.action for s in result.steps}
    assert "search" in actions
    assert "calculator" in actions
