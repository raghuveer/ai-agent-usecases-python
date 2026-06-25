"""Integration test — hits the live local Qwen coder via the gateway.

Builds the real SQLite DB from the bundled seed and asks a counting question.
Gated: runs only when RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os

import pytest

from app import llm, sqlagent
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_sql_agent_round_trip_against_local_qwen():
    settings = get_settings()
    client = llm.build_client(settings)
    conn = sqlagent.build_db()

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    try:
        result = sqlagent.answer(
            "How many customers are there?",
            conn=conn,
            llm_call=llm_call,
        )
    finally:
        conn.close()

    # Model must have produced a valid read-only SELECT (validation passed) ...
    assert result["sql"].upper().startswith("SELECT")
    assert "COUNT" in result["sql"].upper()
    # ... and executing it returns the seeded count of 5 customers.
    assert result["rows"]
    only_value = list(result["rows"][0].values())[0]
    assert int(only_value) == 5
