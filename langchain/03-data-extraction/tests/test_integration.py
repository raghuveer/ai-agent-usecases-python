# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langchain). See langchain/03-data-extraction/README.md
"""Integration test — hits the live local Qwen via the gateway.

Extracts an Invoice from the bundled sample invoice. Gated: skipped unless
``RUN_INTEGRATION=1``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

RUN = os.getenv("RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run"),
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_extract_invoice_against_local_qwen():
    from app.extract import extract_invoice
    from app.llm import build_llm
    from app.settings import get_settings

    settings = get_settings()
    llm = build_llm(settings, max_tokens=512)

    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    invoice = extract_invoice(text, llm, settings.llm_model)

    assert invoice.invoice_number.strip()
    assert invoice.total > 0
