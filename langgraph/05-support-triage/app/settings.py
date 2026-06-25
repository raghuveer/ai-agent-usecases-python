# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC5 Customer support triage (langgraph). See langgraph/05-support-triage/README.md
"""Configuration via pydantic-settings (UC5 support-triage).

Defaults are sensible so the app imports without a real gateway key; unit tests
never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 384
    # Confidence below which a ticket is escalated to a human agent.
    escalate_threshold: float = 0.5


def get_settings() -> Settings:
    return Settings()
