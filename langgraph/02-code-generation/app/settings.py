# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langgraph). See langgraph/02-code-generation/README.md
"""Configuration via pydantic-settings (UC2 code-generation).

Reads the shared env-var contract plus codegen knobs. Defaults are sensible so
the app imports without a real gateway key; unit tests never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-coder"  # gateway alias (free local Qwen coder)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    default_language: str = "python"
    # When 1, the OPTIONAL self-check executes generated python in a subprocess.
    # Off by default so /run never runs untrusted code unless explicitly enabled.
    run_code_check: bool = False
    code_check_timeout: int = 10  # seconds for the subprocess smoke run


def get_settings() -> Settings:
    return Settings()
