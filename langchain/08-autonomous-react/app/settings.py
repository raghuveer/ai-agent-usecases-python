"""Configuration via pydantic-settings (UC8 autonomous-react).

Defaults are sensible so the app imports without a real gateway key; unit tests
never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    # UC8: the two-tool ReAct chain is too unreliable on local Qwen; use Haiku.
    llm_model: str = "claude-haiku-4-5"
    max_steps: int = 6  # default cap on ReAct loop iterations


def get_settings() -> Settings:
    return Settings()
