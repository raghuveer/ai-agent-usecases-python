# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (raw-api). See raw-api/01-rag/README.md
"""LLM client factory for the raw-api approach.

We talk to the gateway with the plain ``openai`` SDK so every byte sent is
explicit. The client is injectable: ``main.py`` builds one at startup and
stores it on app state; unit tests pass a stub so nothing hits the network.

qwen3 ships a "thinking" mode; we disable it by prepending ``/no_think`` to the
system prompt when the model id starts with ``qwen3``.
"""
from __future__ import annotations

from openai import OpenAI

from .settings import Settings


def build_client(settings: Settings) -> OpenAI:
    """Construct an OpenAI SDK client pointed at the gateway."""
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_gateway_key)


def model_profile(model: str) -> dict:
    """Per-model-family quirks/capabilities, keyed by id prefix.

    The single place model-family quirks live. Extend here to support a new
    model family (e.g. a different thinking-mode toggle) -- nothing else changes.
    """
    if model.lower().startswith("qwen3"):
        return {"thinking_prefix": "/no_think\n"}
    return {"thinking_prefix": ""}


def apply_no_think(model: str, system_prompt: str) -> str:
    """Prepend the model's thinking-mode prefix (e.g. ``/no_think`` for qwen3).

    Thin wrapper over :func:`model_profile` so quirks stay in one place.
    """
    return model_profile(model)["thinking_prefix"] + system_prompt


def chat(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 384,
    temperature: float = 0.0,
) -> str:
    """Single chat call. Returns the assistant message text (stripped)."""
    system_prompt = apply_no_think(model, system_prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
