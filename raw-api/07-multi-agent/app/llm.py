# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""LLM client factory for the raw-api approach.

We talk to the gateway with the plain ``openai`` SDK so every byte sent is
explicit. The client is injectable: ``main.py`` builds one at startup and stores
it on app state; unit tests pass a stub so nothing hits the network.

Each sub-agent (researcher LLM step is replaced by deterministic search; writer
and reviewer are role-prompted LLM calls) issues a single system+user chat. We
expose ``chat`` taking a system prompt and a user message.

qwen3 ships a "thinking" mode; we disable it by prepending ``/no_think`` to the
system prompt when the model id starts with ``qwen3``.
"""
from __future__ import annotations

import re
import time
from typing import Callable

from openai import OpenAI

from .settings import Settings


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
    system: str,
    user: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    on_call: Callable[[dict], None] | None = None,
) -> str:
    """Single system+user chat call. Returns assistant text (stripped).

    ``on_call`` receives what actually went over the wire — messages sent,
    completion, token usage, elapsed ms. The tracer's hook; nothing here knows
    what a tracer is.
    """
    messages = [
        {"role": "system", "content": apply_no_think(model, system)},
        {"role": "user", "content": user},
    ]
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    reply = strip_thinking(resp.choices[0].message.content or "")

    if on_call is not None:
        usage = getattr(resp, "usage", None)
        on_call(
            {
                "messages": messages,
                "completion": reply,
                "duration_ms": elapsed_ms,
                "input_tokens": _token_count(usage, "prompt_tokens"),
                "output_tokens": _token_count(usage, "completion_tokens"),
            }
        )
    return reply


def _token_count(usage: object | None, field: str) -> int:
    """A token count, or 0 when absent or non-numeric — never a guess."""
    value = getattr(usage, field, 0)
    return value if isinstance(value, int) else 0


def chat_stream(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
):
    """Yield assistant text incrementally; thinking spans never reach callers."""
    think = ThinkFilter()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": apply_no_think(model, system)},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        if piece:
            visible = think.feed(piece)
            if visible:
                yield visible
    tail = think.flush()
    if tail:
        yield tail
