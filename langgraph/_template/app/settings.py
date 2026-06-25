# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langgraph project template. See langgraph/_template/README.md
"""Configuration via pydantic-settings.

Reads the shared env-var contract. Defaults are sensible so the app imports
without a real gateway key (unit tests never hit the network).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen3:1.7b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512


def get_settings() -> Settings:
    return Settings()
