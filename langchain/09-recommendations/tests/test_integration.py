# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (langchain). See langchain/09-recommendations/README.md
"""Integration test — hits the live local model via the gateway.

Loads the real bundled catalog + profiles, runs the deterministic ranking, and
invokes the LCEL reason chain once per item. Gated: runs only when
RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os

import pytest

from app.llm import build_llm
from app.recommend import load_catalog, load_profiles, recommend
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_recommendations_round_trip_against_local_qwen():
    settings = get_settings()
    llm = build_llm(max_tokens=128)
    catalog = load_catalog()
    profiles = load_profiles()

    result = recommend("u1", catalog, profiles, llm, settings.llm_model, k=3)
    recs = result["recommendations"]
    assert len(recs) == 3
    assert all(r["reason"].strip() for r in recs)
    assert all(r["item_id"] and r["title"] for r in recs)
