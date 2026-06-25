# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (raw-api). See raw-api/04-research-agent/README.md
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
    # Default to the budget Anthropic model. The free local Qwen could not
    # reliably drive the multi-step text ReAct loop (it emitted prose instead of
    # the Action/Action Input protocol and used outside knowledge), so per the
    # build spec's fallback rule this use case switches to claude-haiku-4-5.
    llm_model: str = "claude-haiku-4-5"  # gateway alias (budget Anthropic)
    agent_max_steps: int = 6
    agent_top_k: int = 3


def get_settings() -> Settings:
    return Settings()
