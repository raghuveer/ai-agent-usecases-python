# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""Application settings, loaded from environment / .env via pydantic-settings.

The env var *names* match the other three approaches (`LLM_BASE_URL`,
`LLM_GATEWAY_KEY`, `LLM_MODEL`) so a reader can diff the four approaches without
relearning configuration. ``agent.py`` translates them into the environment
variables the Agent SDK's subprocess actually reads.

Note the base-URL shape differs from the other approaches: the Agent SDK speaks
the **Anthropic** surface (``/v1/messages``) and appends the path itself, so
``LLM_BASE_URL`` has **no** ``/v1`` suffix here. The OpenAI-surface projects use
``http://localhost:8094/v1``; this one uses ``http://localhost:8094``.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Gateway (Anthropic surface — no trailing /v1; the SDK appends /v1/messages).
    llm_base_url: str = "http://localhost:8094"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "claude-haiku"

    # Agent-loop budget caps. These matter far more here than in the other three
    # approaches: an agent loop makes an unbounded number of model calls, so both
    # a turn cap and a hard dollar cap are set on every run.
    agent_max_turns: int = 12
    agent_max_budget_usd: float = 1.00
    agent_effort: str = "low"

    # Web access is OPT-IN. Default offline keeps the example reproducible
    # and runnable on an air-gapped host; set RESEARCH_ALLOW_WEB=1 to enable
    # the built-in WebSearch/WebFetch tools.
    research_allow_web: bool = False


def get_settings() -> Settings:
    return Settings()
