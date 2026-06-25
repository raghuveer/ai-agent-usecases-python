# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langchain project template. See langchain/_template/README.md
"""LLM client factory.

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
The factory is injectable: callers (and tests) can pass their own model in,
so unit tests swap in ``FakeListChatModel`` and never touch the network.

qwen3 has a "thinking" mode that is disabled by prepending ``/no_think`` to the
system prompt. The model-family quirk now lives in ``model_profile`` below (the
single place to extend for a new family); ``app.main`` reads it via that helper.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


def build_llm(settings: Settings | None = None, **kwargs) -> BaseChatModel:
    """Build a ChatOpenAI client against the gateway.

    Parameters are read from ``settings`` (env). Extra ``kwargs`` (e.g.
    ``max_tokens``, ``temperature``) are forwarded to ``ChatOpenAI``.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=kwargs.pop("temperature", settings.llm_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.llm_max_tokens),
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
