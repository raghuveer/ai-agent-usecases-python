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
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    # Confidence below which a ticket is escalated to a human agent.
    escalate_threshold: float = 0.5


def get_settings() -> Settings:
    return Settings()
