"""LLM client factory.

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
The factory is injectable: callers (and tests) pass their own model in, so unit
tests swap in ``FakeListChatModel`` and never touch the network.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


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
        temperature=0,
        **kwargs,
    )


def system_prompt(text: str, model: str) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    if model.startswith("qwen3"):
        return "/no_think\n" + text
    return text
