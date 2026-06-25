"""LLM client factory for the raw-api approach.

We talk to the gateway with the plain ``openai`` SDK so every byte sent is
explicit. The client is injectable: ``main.py`` builds one at startup and
stores it on app state; unit tests pass a stub so nothing hits the network.

The ReAct loop drives a multi-turn conversation, so unlike UC1 this ``chat``
takes a full ``messages`` list rather than a single system+user pair.

qwen3 ships a "thinking" mode; we disable it by prepending ``/no_think`` to the
system prompt when the model id starts with ``qwen3``.
"""
from __future__ import annotations

from openai import OpenAI

from .settings import Settings


def build_client(settings: Settings) -> OpenAI:
    """Construct an OpenAI SDK client pointed at the gateway."""
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_gateway_key)


def apply_no_think(model: str, system_prompt: str) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    if model.startswith("qwen3"):
        return "/no_think\n" + system_prompt
    return system_prompt


def chat(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 384,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> str:
    """Single chat call over a message list. Returns assistant text (stripped).

    The first message is assumed to be the system prompt; ``/no_think`` is
    applied to it for qwen3 models.
    """
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = apply_no_think(model, msgs[0]["content"])
    kwargs: dict = dict(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if stop:
        kwargs["stop"] = stop
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()
