# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langgraph). See langgraph/03-data-extraction/README.md
"""Integration test for UC3 data-extraction (langgraph).

Hits the live local Qwen model via the gateway and extracts the bundled sample
invoice. Skipped unless RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run against the live local model",
)
def test_extract_invoice_against_local_qwen():
    from app.extract import extract_invoice
    from app.llm import build_llm
    from app.settings import get_settings

    settings = get_settings()
    llm = build_llm(settings)  # max_tokens from settings.llm_max_tokens (default 512)

    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    invoice = extract_invoice(text, llm, settings)

    assert invoice.invoice_number.strip()
    assert invoice.total > 0
