# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langchain). See langchain/06-sql-agent/README.md
"""LLM client factory.

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
The factory is injectable: callers (and tests) pass their own model in, so unit
tests swap in ``FakeListChatModel`` and never touch the network.
"""
from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


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


def strip_thinking_stream(chunks):
    """Streaming-safe version of :func:`strip_thinking` for LCEL chains.

    **A plain function in an LCEL pipe silently disables streaming.** LangChain
    wraps it in a ``RunnableLambda``, which must materialise its whole input
    before calling — so ``prompt | llm | StrOutputParser() | strip_thinking``
    yields exactly one chunk per chain, and `chain.stream()` stops being a
    stream at all. Measured here: 2 token frames instead of ~150.

    A *generator* function is treated as a transform instead: it receives the
    upstream iterator and yields as it goes, so streaming survives the step.
    ``invoke()`` still works — LangChain drains the generator.
    """
    think = ThinkFilter()
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else str(chunk)
        visible = think.feed(text)
        if visible:
            yield visible
    tail = think.flush()
    if tail:
        yield tail


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
    """Build a ChatOpenAI client against the gateway.

    Parameters come from ``settings`` (env); extra ``kwargs`` (``max_tokens``,
    ``temperature``, ...) are forwarded to ``ChatOpenAI``.
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


def system_prompt(text: str, model: str) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    return model_profile(model)["thinking_prefix"] + text
