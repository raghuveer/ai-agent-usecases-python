# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (raw-api). See raw-api/03-data-extraction/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Sensible defaults let the app import (and unit tests run) without a real key.
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
    # Generation params (override via LLM_TEMPERATURE / LLM_MAX_TOKENS).
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512  # primary generation budget for this use case
    # Structured-output strategy: "text" (prompt for JSON, parse from the reply —
    # portable across any chat model) or "native" (provider JSON mode, more
    # reliable but requires provider support). Default reproduces today's behavior.
    llm_structured_mode: str = "text"


def get_settings() -> Settings:
    return Settings()
