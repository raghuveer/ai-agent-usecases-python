# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""Integration test — drives the real multi-agent pipeline via the gateway.

researcher (deterministic) → writer (LLM) → reviewer (LLM) over a bundled topic.
Gated: runs only when RUN_INTEGRATION=1.

UC7 defaults to ``claude-haiku-4-5`` (see README "Model note"): the orchestrator
needs reliable role-following the free local Qwen could not give, so the test is
also marked ``anthropic``.
"""
from __future__ import annotations

import os

import pytest

from app import agents, llm
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_pipeline_against_gateway():
    settings = get_settings()
    client = llm.build_client(settings)

    def llm_call(system: str, user: str) -> str:
        return llm.chat(client, model=settings.llm_model, system=system, user=user)

    result = agents.orchestrate(
        "tidal energy",
        llm_call=llm_call,
        research_top_k=settings.research_top_k,
        max_revisions=settings.max_revisions,
    )

    assert result.draft.strip()
    assert result.review.strip()
    assert isinstance(result.approved, bool)
    # The deterministic researcher fed real corpus facts to the writer.
    assert "tidal" in result.contributions["research"].lower()
