# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC3 Data extraction (langgraph). See langgraph/03-data-extraction/README.md
"""Configuration via pydantic-settings (UC3 data-extraction).

Reads the shared env-var contract. Defaults are sensible so the app imports
without a real gateway key; unit tests never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "claude-haiku"  # UC3: local Qwen too weak for reliable JSON; use Haiku
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    # Structured-output mode: "text" (prompt-for-JSON, parse the reply — portable
    # across any chat model) or "native" (llm.with_structured_output(Invoice) —
    # more reliable but requires provider support). Default reproduces today's
    # text-parsing behavior exactly.
    llm_structured_mode: str = "text"


def get_settings() -> Settings:
    return Settings()
