"""Integration test — drives the real ReAct StateGraph via the gateway.

Two-tool chain: search Northwind's return window (30) then calculator doubles it
(60). Gated by RUN_INTEGRATION=1.

Defaults to ``claude-haiku-4-5`` (see README "Model note"): the free local Qwen
could not reliably drive the two-tool text-ReAct cycle, so this is marked
``anthropic`` too.
"""
from __future__ import annotations

import os

import pytest

from app.llm import build_llm
from app.react import run_react
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_two_tool_cycle_against_gateway():
    settings = get_settings()
    llm = build_llm(settings)
    result = run_react(
        "Find Northwind's return window in days, then tell me what double that is.",
        llm=llm,
        max_steps=settings.max_steps,
    )

    assert result["answer"].strip()
    assert "60" in result["answer"]
    actions = {s["action"] for s in result["steps"]}
    assert "search" in actions
    assert "calculator" in actions
