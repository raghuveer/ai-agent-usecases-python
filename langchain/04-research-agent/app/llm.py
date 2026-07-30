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

STOP_MARKERS = ("\nObservation:", "Observation:")


def build_llm(settings: Settings | None = None, **kwargs) -> BaseChatModel:
    """Build a ChatOpenAI client against the gateway.

    Generation halts at the next ``Observation:`` so the model cannot hallucinate
    tool output — the ReAct loop supplies real observations. ``stop`` is sent only
    where the endpoint supports it; :func:`truncate_at_stop` enforces the cut on
    the reply either way.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=kwargs.pop("temperature", settings.llm_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.llm_max_tokens),
        stop=stop_sequences(settings.llm_model),
        **kwargs,
    )


def model_profile(model: str) -> dict:
    """Per-model-family quirks/capabilities, keyed by id prefix.

    Single place model-family quirks live — extend here to support a new model
    family and nothing else changes.

    ``supports_stop`` is False for ``claude-*``: those aliases route through the
    gateway's Anthropic surface, which does not translate an OpenAI ``stop``
    array into Anthropic's ``stop_sequences`` and answers 500 instead.
    """
    m = model.lower()
    if m.startswith("qwen3"):
        return {"thinking_prefix": "/no_think\n", "supports_stop": True}
    if m.startswith("claude"):
        return {"thinking_prefix": "", "supports_stop": False}
    return {"thinking_prefix": "", "supports_stop": True}


def stop_sequences(model: str) -> list[str] | None:
    """``STOP_MARKERS`` when the endpoint honours ``stop``, else ``None``."""
    return list(STOP_MARKERS) if model_profile(model)["supports_stop"] else None


def truncate_at_stop(text: str, markers: tuple[str, ...] = STOP_MARKERS) -> str:
    """Cut ``text`` at the earliest stop marker.

    Server-side ``stop`` is an optimisation, not a guarantee: some endpoints
    ignore it and the gateway's Anthropic path rejects it outright. Enforcing the
    cut here keeps the loop's invariant -- the model never supplies its own
    Observation -- whatever the endpoint does.
    """
    cut = len(text)
    for marker in markers:
        found = text.find(marker)
        if found != -1:
            cut = min(cut, found)
    return text[:cut].strip()


def system_prompt(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    settings = settings or get_settings()
    return model_profile(settings.llm_model)["thinking_prefix"] + text
