"""Integration test — hits the live local coder model via the gateway.

Generates real code with the free local Qwen coder. Gated: runs only when
RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os

import pytest

from app import codegen, llm
from app.settings import get_settings

pytestmark = pytest.mark.integration

RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_codegen_against_local_qwen():
    settings = get_settings()
    client = llm.build_client(settings)

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    result = codegen.generate(
        "Write a Python function add(a,b) that returns their sum",
        language="python",
        llm_call=llm_call,
        run_code_check=False,
    )

    assert result["language"] == "python"
    assert result["code"].strip()
    assert "def add" in result["code"]
