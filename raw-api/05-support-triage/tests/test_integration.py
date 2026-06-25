# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (raw-api). See raw-api/05-support-triage/README.md
"""Integration test — hits the live local model via the gateway.

Classifies a real support message with the live Qwen model. Gated: runs only
when RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os

import pytest

from app import llm, triage
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_billing_message_classifies_against_local_qwen():
    triage.reset_memory()
    settings = get_settings()
    client = llm.build_client(settings)

    def classify_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=64,
        )

    def respond_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
        )

    result = triage.triage(
        "I was double charged on my invoice",
        session_id="it-1",
        classify_call=classify_call,
        respond_call=respond_call,
        escalate_threshold=settings.escalate_threshold,
    )

    assert result["intent"] == "billing"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["response"].strip()
