# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langchain project template. See langchain/_template/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Defaults are sensible so the app imports without a real gateway key. Unit tests
never need network access, so the placeholder key is fine here.
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
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 256  # primary generation budget (was hardcoded at call site)

    # Carried in the template so use cases that copy it have the same contract.
    rag_top_k: int = 3
    chroma_dir: str = ".chroma"


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads env each call)."""
    return Settings()
