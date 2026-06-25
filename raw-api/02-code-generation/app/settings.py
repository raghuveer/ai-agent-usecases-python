# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (raw-api). See raw-api/02-code-generation/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Sensible defaults let the app import (and unit tests run) without a real key.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-coder"  # gateway alias (free local Qwen coder)
    default_language: str = "python"
    # When 1, the OPTIONAL self-check executes generated python in a subprocess.
    # Off by default so /run never runs untrusted code unless explicitly enabled.
    run_code_check: bool = False
    code_check_timeout: int = 10  # seconds for the subprocess smoke run


def get_settings() -> Settings:
    return Settings()
