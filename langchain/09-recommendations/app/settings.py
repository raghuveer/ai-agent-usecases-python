# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (langchain). See langchain/09-recommendations/README.md
"""Configuration via pydantic-settings (UC9 recommendations).

Reads the shared env-var contract plus the top-k knob. Defaults are sensible so
the app imports without a real gateway key; unit tests never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    llm_temperature: float = 0.0  # deterministic by default
    llm_max_tokens: int = 128  # primary generation budget (was hardcoded at call site)
    rec_top_k: int = 3


def get_settings() -> Settings:
    return Settings()
