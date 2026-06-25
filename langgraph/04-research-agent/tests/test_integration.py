# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (langgraph). See langgraph/04-research-agent/README.md
"""Integration test — drives the REAL StateGraph ReAct loop via the gateway.

Gated: skipped unless ``RUN_INTEGRATION=1``. Also marked ``anthropic`` because
this use case defaults to ``claude-haiku-4-5`` (the free local Qwen could not
reliably drive the multi-step ReAct loop — see README "Model choice").
"""
from __future__ import annotations

import os

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anthropic,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]


def test_research_agent_against_gateway():
    from app import agent
    from app.llm import build_llm
    from app.settings import get_settings

    settings = get_settings()
    llm = build_llm(settings)

    result = agent.run_agent(
        "What is Northwind's return window and what does the warranty cover?",
        corpus=agent.Corpus(),
        llm=llm,
        max_steps=settings.agent_max_steps,
        top_k=settings.agent_top_k,
    )

    assert isinstance(result.answer, str) and result.answer.strip()
    assert result.stopped_reason == "final_answer"
    assert "returns.md" in result.sources
    assert "warranty.md" in result.sources
