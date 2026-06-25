"""LLM client factory (UC8 autonomous-react, langchain approach).

Returns a langchain ``ChatOpenAI`` pointed at the OpenAI-compatible gateway.
Injectable: unit tests swap in ``FakeListChatModel`` and never touch the network.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


def build_llm(settings: Settings | None = None, **kwargs) -> BaseChatModel:
    """Build a ChatOpenAI client against the gateway."""
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=kwargs.pop("max_tokens", 384),
        **kwargs,
    )


def system_prefix(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable thinking mode."""
    settings = settings or get_settings()
    if settings.llm_model.startswith("qwen3"):
        return "/no_think\n" + text
    return text
