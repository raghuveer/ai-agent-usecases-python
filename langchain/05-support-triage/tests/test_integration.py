# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langchain). See langchain/05-support-triage/README.md
"""Integration test — hits the live local Qwen via the gateway.

Gated: skipped unless ``RUN_INTEGRATION=1``. Keep ``max_tokens`` small.
"""
from __future__ import annotations

import os

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]


def test_billing_message_classifies_against_local_qwen():
    from app import triage
    from app.llm import build_llm
    from app.settings import get_settings

    triage.reset_memory()
    settings = get_settings()
    llm = build_llm(settings, max_tokens=256)

    result = triage.triage(
        "I was double charged on my invoice",
        session_id="it-1",
        llm=llm,
        escalate_threshold=settings.escalate_threshold,
    )

    assert result["intent"] == "billing"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["response"], str) and result["response"].strip()
