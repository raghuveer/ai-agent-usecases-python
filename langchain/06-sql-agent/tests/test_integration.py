# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langchain). See langchain/06-sql-agent/README.md
"""Integration test — hits the live local Qwen coder via the gateway.

Builds the real SQLite DB from the bundled seed and asks a counting question.
Gated: skipped unless ``RUN_INTEGRATION=1``.
"""
from __future__ import annotations

import os

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]


def test_sql_agent_roundtrip_against_local_qwen():
    from app.llm import build_llm
    from app.settings import get_settings
    from app.sqlagent import answer_question, build_db

    settings = get_settings()
    llm = build_llm(settings, max_tokens=256)
    conn = build_db()

    try:
        result = answer_question(
            "How many customers are there?", conn, llm, settings.llm_model
        )
    finally:
        conn.close()

    assert result["sql"].upper().startswith("SELECT")
    assert "COUNT" in result["sql"].upper()
    assert result["rows"]
    only_value = list(result["rows"][0].values())[0]
    assert int(only_value) == 5
