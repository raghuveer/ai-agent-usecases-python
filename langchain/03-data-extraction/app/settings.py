# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langchain). See langchain/03-data-extraction/README.md
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
    llm_model: str = "claude-haiku"  # UC3: local Qwen too weak for reliable JSON; use Haiku
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 512  # primary generation budget (was hardcoded at call site)
    # Structured-output strategy: "text" (prompt JSON + parse, portable default)
    # or "native" (llm.with_structured_output; needs provider support).
    llm_structured_mode: str = "text"


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads env each call)."""
    return Settings()
