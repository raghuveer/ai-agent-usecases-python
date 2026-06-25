# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC1 Q&A / RAG chatbot (langgraph). See langgraph/01-rag/README.md
"""Configuration via pydantic-settings (UC1 RAG).

Reads the shared env-var contract plus RAG knobs. Defaults are sensible so the
app imports without a real gateway key; unit tests never touch the network.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:8080/v1"
    llm_gateway_key: str = "replace-with-platform-virtual-key"
    llm_model: str = "qwen-local-instruct"  # gateway alias (free local Qwen)
    rag_top_k: int = 3
    chroma_dir: str = ".chroma"


def get_settings() -> Settings:
    return Settings()
