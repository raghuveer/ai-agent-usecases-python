# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langgraph). See langgraph/08-autonomous-react/README.md
"""LLM client factory (UC8 autonomous-react, langgraph approach).

Uniform ``ChatOpenAI`` path pointed at the OpenAI-compatible gateway, for both
Qwen and Claude. Injectable so unit tests pass a ``FakeListChatModel``.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


def model_profile(model: str) -> dict:
    """Per-model-family quirks/capabilities, keyed by id prefix.

    Single place model-family quirks live. Extend here to support a new model
    family — nothing else changes.
    """
    if model.lower().startswith("qwen3"):
        return {"thinking_prefix": "/no_think\n"}
    return {"thinking_prefix": ""}


def build_llm(settings: Settings | None = None, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at the gateway."""
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=kwargs.pop("max_tokens", settings.llm_max_tokens),
        **kwargs,
    )


def system_prefix(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode.

    Thin wrapper over :func:`model_profile` — the single source of model quirks.
    """
    settings = settings or get_settings()
    return model_profile(settings.llm_model)["thinking_prefix"] + text
