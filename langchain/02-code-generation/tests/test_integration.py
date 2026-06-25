# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langchain). See langchain/02-code-generation/README.md
"""Integration test — hits the live local coder model via the gateway.

Generates real code with the free local Qwen coder. Skipped unless
RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]


def test_codegen_against_local_qwen():
    from app.codegen import generate
    from app.llm import build_llm
    from app.settings import get_settings

    settings = get_settings()
    llm = build_llm(settings, max_tokens=512)

    result = generate(
        "Write a Python function add(a,b) that returns their sum",
        language="python",
        llm=llm,
        model_name=settings.llm_model,
        run_code_check=False,
    )

    assert result["language"] == "python"
    assert result["code"].strip()
    assert "def add" in result["code"]
