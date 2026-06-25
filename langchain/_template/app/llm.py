"""LLM client factory.

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
The factory is injectable: callers (and tests) can pass their own model in,
so unit tests swap in ``FakeListChatModel`` and never touch the network.

qwen3 has a "thinking" mode that is disabled by prepending ``/no_think`` to the
system prompt — see ``app.main`` / use-case logic, not here, because that is a
prompt concern, not a client concern.
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
        temperature=0,
        **kwargs,
    )
