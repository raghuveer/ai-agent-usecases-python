"""Integration test — drives the REAL text ReAct loop via the gateway.

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
    from app import agent, llm
    from app.settings import get_settings

    settings = get_settings()
    client = llm.build_client(settings)

    def llm_call(user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=agent.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    result = agent.run_agent(
        "What is Northwind's return window and what does the warranty cover?",
        corpus=agent.Corpus(),
        llm_call=llm_call,
        max_steps=settings.agent_max_steps,
        top_k=settings.agent_top_k,
    )

    assert isinstance(result.answer, str) and result.answer.strip()
    assert result.stopped_reason == "final_answer"
    # The agent must have searched and surfaced both topics' sources.
    assert "returns.md" in result.sources
    assert "warranty.md" in result.sources
