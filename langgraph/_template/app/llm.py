"""LLM client factory.

One uniform client path (langchain-openai ``ChatOpenAI`` pointed at the gateway)
for both Qwen and Claude, so the cross-approach comparison stays honest. The
factory is injectable: tests pass their own fake chat model instead.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .settings import Settings, get_settings


def build_llm(settings: Settings | None = None, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at the OpenAI-compatible gateway."""
    settings = settings or get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_gateway_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=512,
        **kwargs,
    )


def system_prompt(text: str, settings: Settings | None = None) -> str:
    """Prepend ``/no_think`` for qwen3 models to disable the thinking mode."""
    settings = settings or get_settings()
    if settings.llm_model.startswith("qwen3"):
        return "/no_think\n" + text
    return text
