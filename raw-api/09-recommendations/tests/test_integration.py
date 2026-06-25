# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (raw-api). See raw-api/09-recommendations/README.md
"""Integration test — hits the live local model via the gateway.

Loads the real bundled catalog + profiles, runs the deterministic ranking, and
calls the LLM for one grounded reason per item. Gated: runs only when
RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os

import pytest

from app import llm, recommend
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_recommendations_round_trip_against_local_qwen():
    settings = get_settings()
    client = llm.build_client(settings)

    catalog = recommend.load_catalog()
    profiles = recommend.load_profiles()

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    result = recommend.recommend(
        "u1",
        catalog=catalog,
        profiles=profiles,
        llm_call=llm_call,
        k=3,
    )

    recs = result["recommendations"]
    assert len(recs) == 3
    assert all(r["reason"].strip() for r in recs)
    assert all(r["item_id"] and r["title"] for r in recs)
