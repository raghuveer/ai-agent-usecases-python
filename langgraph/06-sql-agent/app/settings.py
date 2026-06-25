# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langgraph). See langgraph/06-sql-agent/README.md
"""Configuration via pydantic-settings (UC6 sql-agent).

Reads the shared env-var contract. Defaults are sensible so the app imports
without a real gateway key; unit tests never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-coder"  # gateway alias (free local Qwen coder)


def get_settings() -> Settings:
    return Settings()
