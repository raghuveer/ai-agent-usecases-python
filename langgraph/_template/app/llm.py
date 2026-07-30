# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langgraph project template. See langgraph/_template/README.md
"""LLM client factory.

One uniform client path (langchain-openai ``ChatOpenAI`` pointed at the gateway)
for both Qwen and Claude, so the cross-approach comparison stays honest. The
factory is injectable: tests pass their own fake chat model instead.
"""
from __future__ import annotations

import re

from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks from a reply.

    ``/no_think`` tells qwen3 not to *reason* in the block, but the model still
    emits an empty pair of tags — which then appear verbatim in the API
    response. Caught by running the Docker quickstart, whose default model is a
    qwen3 tag; the gateway's ``qwen-local-instruct`` alias (qwen2.5) has no
    thinking mode, so this never showed up in the usual configuration.

    Reasoning is stripped rather than returned: it is not the answer, and
    echoing a model's chain-of-thought to callers is a bad default.
    """
    return _THINK_BLOCK.sub("", text or "").strip()


def model_profile(model: str) -> dict:
    """Per-model-family quirks/capabilities, keyed by id prefix.

    Single place model-family quirks live. Extend here to support a new model
    family — nothing else changes.
    """
    if model.lower().startswith("qwen3"):
        return {"thinking_prefix": "/no_think\n"}
    return {"thinking_prefix": ""}


def build_llm(settings: Settings | None = None, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at the OpenAI-compatible gateway.

    Generation params come from settings (env-configurable); ``**kwargs`` still
    override them.
    """
    settings = settings or get_settings()
    params = {
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_gateway_key,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


def system_prompt(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable the thinking mode.

    Thin wrapper over :func:`model_profile` — the single source of model quirks.
    """
    settings = settings or get_settings()
    return model_profile(settings.llm_model)["thinking_prefix"] + text
