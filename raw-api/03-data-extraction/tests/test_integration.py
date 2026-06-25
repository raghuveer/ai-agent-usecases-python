"""Integration test — hits the live local model via the gateway.

Extracts an Invoice from the bundled sample invoice. Gated: runs only when
RUN_INTEGRATION=1, otherwise skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app import extract, llm
from app.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anthropic]

RUN = os.getenv("RUN_INTEGRATION") == "1"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(not RUN, reason="set RUN_INTEGRATION=1 to run live-model tests")
def test_extract_invoice_against_local_qwen():
    settings = get_settings()
    client = llm.build_client(settings)

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        return llm.chat(
            client,
            model=settings.llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    text = (DATA_DIR / "sample-invoice.txt").read_text(encoding="utf-8")
    invoice = extract.extract_invoice(text, llm_call=llm_call)

    # The model should populate the key fields from the sample invoice.
    assert invoice.invoice_number.strip()
    assert invoice.total > 0
