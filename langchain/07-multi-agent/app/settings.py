# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langchain). See langchain/07-multi-agent/README.md
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

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    # UC7: coordinating writer + reviewer reliably is beyond the free local Qwen.
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 400  # primary generation budget (was hardcoded in build_llm)
    research_top_k: int = 4  # bullet facts the researcher gathers from the corpus
    max_revisions: int = 1  # cap on the reviewer reject → writer revise loop


def get_settings() -> Settings:
    return Settings()
