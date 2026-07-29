# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — raw-api project template. See raw-api/_template/README.md
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
    llm_model: str = "qwen-local-instruct"
    # Generation params (override via LLM_TEMPERATURE / LLM_MAX_TOKENS).
    llm_temperature: float = 0.0
    llm_max_tokens: int = 128  # primary generation budget for this use case
    rag_top_k: int = 3
    chroma_dir: str = ".chroma"


def get_settings() -> Settings:
    return Settings()
