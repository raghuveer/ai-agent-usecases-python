# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Sensible defaults let the app import (and unit tests run) without a real key.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    # UC8: the two-tool ReAct chain is too unreliable on local Qwen; use Haiku.
    llm_model: str = "claude-haiku-4-5"
    # Generation params (override via LLM_TEMPERATURE / LLM_MAX_TOKENS).
    llm_temperature: float = 0.0
    llm_max_tokens: int = 384  # primary generation budget for this use case
    max_steps: int = 6  # default cap on ReAct loop iterations


def get_settings() -> Settings:
    return Settings()
