# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langchain). See langchain/08-autonomous-react/README.md
"""LLM client factory (UC8 autonomous-react, langchain approach).

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
Injectable: unit tests swap in ``FakeListChatModel`` and never touch the network.
"""
from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings

STOP_MARKERS = ("Observation:",)


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class ThinkFilter:
    """Drops ``<think>…</think>`` spans from a *token stream*, incrementally.

    Streaming makes the thinking-tag problem harder than it looks. In a complete
    reply you can regex the block out; in a stream the tags arrive split across
    chunks (``"<th"`` + ``"ink>"``), and by the time you recognise one you may
    already have forwarded its contents to the client.

    So text is held back whenever it could still turn out to be a tag: anything
    after a ``<`` is buffered until it either completes a tag or proves not to
    be one. That costs a few characters of latency and is the only way to
    guarantee reasoning never reaches the caller.
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        """Return the part of ``chunk`` that is safe to emit now."""
        self._buffer += chunk
        out: list[str] = []

        while self._buffer:
            if self._inside:
                end = self._buffer.find(self._CLOSE)
                if end == -1:
                    # Keep only enough to recognise a close tag split across chunks.
                    self._buffer = self._buffer[-(len(self._CLOSE) - 1):]
                    break
                self._buffer = self._buffer[end + len(self._CLOSE):]
                self._inside = False
                continue

            start = self._buffer.find(self._OPEN)
            if start != -1:
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self._OPEN):]
                self._inside = True
                continue

            # No complete open tag. Emit everything that cannot be the start of
            # one; hold back a possible partial tag at the tail.
            cut = self._buffer.rfind("<")
            if cut == -1:
                out.append(self._buffer)
                self._buffer = ""
            else:
                out.append(self._buffer[:cut])
                self._buffer = self._buffer[cut:]
            break

        return "".join(out)

    def flush(self) -> str:
        """Emit whatever is held back, at end of stream."""
        tail = "" if self._inside else self._buffer
        self._buffer = ""
        return tail


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


def build_llm(settings: Settings | None = None, **kwargs) -> BaseChatModel:
    """Build a ChatOpenAI client against the gateway."""
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


def stop_sequences(llm=None, settings: Settings | None = None) -> list[str] | None:
    """``STOP_MARKERS`` when the endpoint honours ``stop``, else ``None``.

    The model id comes from the client where it exposes one (a test double may
    not), otherwise from settings.
    """
    model = getattr(llm, "model_name", None) or (settings or get_settings()).llm_model
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


def system_prefix(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    settings = settings or get_settings()
    return model_profile(settings.llm_model)["thinking_prefix"] + text
