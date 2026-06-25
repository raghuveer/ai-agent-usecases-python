# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (langchain). See langchain/09-recommendations/README.md
"""Recommendations core (langchain): deterministic ranking + an LCEL reason chain.

Responsibilities are split deliberately:

- **Ranking is plain Python** (``rank_items``): score each catalog item by overlap
  with the user's liked genres/tags, sort deterministically, take top-k. No model
  is involved, so results are reproducible and explainable.
- **Only the explanation uses the LLM**, via a small LCEL chain:

      prompt | llm | StrOutputParser

  ``build_reason_chain`` returns that runnable; ``recommend`` invokes it once per
  recommended item to produce a grounded one-sentence reason.

The ranking functions and the chain take their dependencies as arguments, so unit
tests inject a ``FakeListChatModel`` and run fully offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import model_profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = (
    "You are a recommendation assistant. In ONE short sentence, explain why the "
    "given item fits the user's tastes. Ground the reason ONLY in the user's "
    "liked genres/tags and the item's genres/tags provided. Do not invent facts."
)

USER_PROMPT = (
    "User {name} likes genres {liked_genres} and tags {liked_tags}.\n"
    "Item '{title}' has genres {genres} and tags {tags}.\n"
    "Why does this item fit the user? Answer in one sentence."
)


@dataclass
class Item:
    id: str
    title: str
    genres: list[str]
    tags: list[str]


@dataclass
class Profile:
    user_id: str
    name: str
    liked_genres: list[str]
    liked_tags: list[str]


@dataclass
class Scored:
    item: Item
    score: int


def _system_prompt(model: str) -> str:
    """Disable qwen3 thinking mode by prefixing /no_think."""
    return model_profile(model)["thinking_prefix"] + SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_catalog(data_dir: Path | None = None) -> list[Item]:
    data_dir = data_dir or DATA_DIR
    raw = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    return [
        Item(id=r["id"], title=r["title"], genres=r["genres"], tags=r["tags"])
        for r in raw
    ]


def load_profiles(data_dir: Path | None = None) -> dict[str, Profile]:
    data_dir = data_dir or DATA_DIR
    raw = json.loads((data_dir / "profiles.json").read_text(encoding="utf-8"))
    return {
        r["user_id"]: Profile(
            user_id=r["user_id"],
            name=r["name"],
            liked_genres=r["liked_genres"],
            liked_tags=r["liked_tags"],
        )
        for r in raw
    }


# --------------------------------------------------------------------------- #
# Deterministic ranking (plain Python, no LLM)
# --------------------------------------------------------------------------- #
def score_item(item: Item, profile: Profile) -> int:
    """Overlap count between the item and the profile's liked genres/tags."""
    liked_genres = set(profile.liked_genres)
    liked_tags = set(profile.liked_tags)
    return len(set(item.genres) & liked_genres) + len(set(item.tags) & liked_tags)


def rank_items(catalog: list[Item], profile: Profile, k: int) -> list[Scored]:
    """Rank by overlap score (desc), tie-break by item id (asc); take top-k.

    Items with zero overlap are dropped — we only recommend genuine matches.
    """
    scored = [Scored(item=it, score=score_item(it, profile)) for it in catalog]
    scored = [s for s in scored if s.score > 0]
    scored.sort(key=lambda s: (-s.score, s.item.id))
    return scored[:k]


# --------------------------------------------------------------------------- #
# LLM explanation chain (only the "why" sentence)
# --------------------------------------------------------------------------- #
def build_reason_chain(llm: BaseChatModel, model_name: str) -> Runnable:
    """Assemble the LCEL chain (item/profile fields -> one-sentence reason)."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_prompt(model_name)), ("human", USER_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


def recommend(
    user_id: str,
    catalog: list[Item],
    profiles: dict[str, Profile],
    llm: BaseChatModel,
    model_name: str,
    k: int,
) -> dict:
    """Full pass: profile lookup -> deterministic rank -> LCEL reason per item."""
    profile = profiles.get(user_id)
    if profile is None:
        raise KeyError(user_id)

    chain = build_reason_chain(llm, model_name)
    top = rank_items(catalog, profile, k)
    recommendations = []
    for s in top:
        reason = chain.invoke(
            {
                "name": profile.name,
                "liked_genres": profile.liked_genres,
                "liked_tags": profile.liked_tags,
                "title": s.item.title,
                "genres": s.item.genres,
                "tags": s.item.tags,
            }
        )
        recommendations.append(
            {"item_id": s.item.id, "title": s.item.title, "reason": reason}
        )
    return {"recommendations": recommendations}
