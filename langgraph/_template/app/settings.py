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


def get_settings() -> Settings:
    return Settings()
