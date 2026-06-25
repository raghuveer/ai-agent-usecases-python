"""Integration test — drives the real langchain pipeline via the gateway.

researcher (deterministic) → writer chain → reviewer chain over a bundled topic.
Gated: runs only when RUN_INTEGRATION=1.

UC7 defaults to ``claude-haiku-4-5`` (see README "Model note"): role-following the
free local Qwen could not give, so the test is also marked ``anthropic``.
"""
from __future__ import annotations

import os

import pytest

from app import agents
from app.llm import build_llm
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_pipeline_against_gateway():
    settings = get_settings()
    llm = build_llm(settings)

    result = agents.orchestrate(
        "tidal energy",
        llm=llm,
        research_top_k=settings.research_top_k,
        max_revisions=settings.max_revisions,
    )

    assert result.draft.strip()
    assert result.review.strip()
    assert isinstance(result.approved, bool)
    assert "tidal" in result.contributions["research"].lower()
