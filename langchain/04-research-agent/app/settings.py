# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (langchain). See langchain/04-research-agent/README.md
"""Configuration via pydantic-settings (UC4 research-agent).

Defaults are sensible so the app imports without a real gateway key; unit tests
never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    # Default to the budget Anthropic model. The free local Qwen could not
    # reliably drive the multi-step text ReAct loop (it emitted prose instead of
    # the Action/Action Input protocol and used outside knowledge), so per the
    # build spec's fallback rule this use case switches to claude-haiku.
    llm_model: str = "claude-haiku"  # gateway alias (budget Anthropic)
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 512  # primary generation budget (was hardcoded in build_llm)
    agent_max_steps: int = 6
    agent_top_k: int = 3


def get_settings() -> Settings:
    return Settings()
