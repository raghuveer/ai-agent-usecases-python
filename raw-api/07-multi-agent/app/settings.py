# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

Sensible defaults let the app import (and unit tests run) without a real key.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    # UC7: the orchestrator → researcher → writer → reviewer chain needs reliable
    # role-following the free local Qwen can't give; use Haiku.
    llm_model: str = "claude-haiku"
    # Generation params (override via LLM_TEMPERATURE / LLM_MAX_TOKENS).
    llm_temperature: float = 0.0
    llm_max_tokens: int = 400  # primary generation budget for this use case
    research_top_k: int = 4  # bullet facts the researcher gathers from the corpus
    max_revisions: int = 1  # cap on the reviewer reject → writer revise loop

    # Tracing (see docs/trace-format.md). `?trace=1` always returns the trace
    # inline; the sink controls whether anything is written to disk.
    trace_sink: str = "none"  # none | file
    trace_dir: str = "traces"
    # Traces embed the full prompt, so they contain whatever the caller sent.
    # Set 0 to keep timings and token counts but drop message content.
    trace_include_prompts: bool = True


def get_settings() -> Settings:
    return Settings()
