# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langgraph). See langgraph/07-multi-agent/README.md
"""Integration test — drives the real multi-agent StateGraph via the gateway.

researcher (deterministic) → writer node → reviewer node over a bundled topic.
Gated: runs only when RUN_INTEGRATION=1.

UC7 defaults to ``claude-haiku-4-5`` (see README "Model note"): coordinating the
sub-agents reliably is beyond the free local Qwen, so the test is also marked
``anthropic``.
"""
from __future__ import annotations

import os

import pytest

from app.graph import run_multi_agent
from app.llm import build_llm
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_graph_against_gateway():
    settings = get_settings()
    llm = build_llm(settings)

    result = run_multi_agent(
        "tidal energy",
        llm=llm,
        research_top_k=settings.research_top_k,
        max_revisions=settings.max_revisions,
    )

    assert result["draft"].strip()
    assert result["review"].strip()
    assert isinstance(result["approved"], bool)
    assert "tidal" in result["contributions"]["research"].lower()
