# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langgraph). See langgraph/10-hitl-approval/README.md
"""Configuration via pydantic-settings (UC10 hitl-approval).

Defaults are sensible so the app imports without a real gateway key; unit tests
never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8094/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256

    # Tracing (see docs/trace-format.md). `?trace=1` always returns the trace
    # inline; the sink controls whether anything is written to disk.
    trace_sink: str = "none"  # none | file
    trace_dir: str = "traces"
    # Traces embed the full prompt, so they contain whatever the caller sent.
    # Set 0 to keep timings and token counts but drop message content.
    trace_include_prompts: bool = True


def get_settings() -> Settings:
    return Settings()
