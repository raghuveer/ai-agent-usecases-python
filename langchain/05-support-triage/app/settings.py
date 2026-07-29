# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langchain). See langchain/05-support-triage/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Defaults are sensible so the app imports without a real gateway key. Unit tests
never need network access.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 384  # primary generation budget (was hardcoded at call site)
    # Confidence below which a ticket is escalated to a human agent.
    escalate_threshold: float = 0.5


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads env each call)."""
    return Settings()
