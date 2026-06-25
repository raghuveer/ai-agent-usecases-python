# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (langchain). See langchain/04-research-agent/README.md
"""LLM client factory (UC4 research-agent, langchain approach).

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway. The
factory is injectable: unit tests pass a ``FakeListChatModel`` and never touch
the network.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


def build_llm(settings: Settings | None = None, **kwargs) -> BaseChatModel:
    """Build a ChatOpenAI client against the gateway.

    ``stop`` halts generation at the next ``Observation:`` so the model cannot
    hallucinate tool output — the ReAct loop supplies real observations.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=kwargs.pop("temperature", settings.llm_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.llm_max_tokens),
        stop=["\nObservation:", "Observation:"],
        **kwargs,
    )


def model_profile(model: str) -> dict:
    """Per-model-family quirks/capabilities, keyed by id prefix.

    Single place model-family quirks live — extend here to support a new model
    family and nothing else changes.
    """
    if model.lower().startswith("qwen3"):
        return {"thinking_prefix": "/no_think\n"}
    return {"thinking_prefix": ""}


def system_prompt(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    settings = settings or get_settings()
    return model_profile(settings.llm_model)["thinking_prefix"] + text
