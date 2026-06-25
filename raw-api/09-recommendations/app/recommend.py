# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (raw-api). See raw-api/09-recommendations/README.md
"""Hand-written recommendations: deterministic ranking + LLM-written reasons.

The point of this use case is the split of responsibilities:

- **Ranking is plain Python** (``rank_items``). We score each catalog item by
  how many of its genres/tags overlap the user's liked genres/tags, then sort
  deterministically (score desc, then id asc as a stable tie-break) and take the
  top-k. No model is involved, so the recommendations are reproducible.
- **Only the explanation uses the LLM** (``reason_for``). For each chosen item we
  ask the model for ONE grounded sentence saying why it fits the profile.

Both data loading and the LLM call are injectable so unit tests run fully
offline with no network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Keep the reason short and grounded; the model only fills in the "why".
SYSTEM_PROMPT = (
    "You are a recommendation assistant. In ONE short sentence, explain why the "
    "given item fits the user's tastes. Ground the reason ONLY in the user's "
    "liked genres/tags and the item's genres/tags provided. Do not invent facts."
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


# An LLM call is `(system_prompt, user_prompt) -> reason`. Injectable for tests.
LLMCall = Callable[[str, str], str]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_catalog(data_dir: Path = DATA_DIR) -> list[Item]:
    raw = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    return [
        Item(id=r["id"], title=r["title"], genres=r["genres"], tags=r["tags"])
        for r in raw
    ]


def load_profiles(data_dir: Path = DATA_DIR) -> dict[str, Profile]:
    raw = json.loads((data_dir / "profiles.json").read_text(encoding="utf-8"))
    profiles = {}
    for r in raw:
        profiles[r["user_id"]] = Profile(
            user_id=r["user_id"],
            name=r["name"],
            liked_genres=r["liked_genres"],
            liked_tags=r["liked_tags"],
        )
    return profiles


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
# LLM explanation (only the "why" sentence)
# --------------------------------------------------------------------------- #
def build_reason_prompt(item: Item, profile: Profile) -> str:
    return (
        f"User {profile.name} likes genres {profile.liked_genres} and "
        f"tags {profile.liked_tags}.\n"
        f"Item '{item.title}' has genres {item.genres} and tags {item.tags}.\n"
        "Why does this item fit the user? Answer in one sentence."
    )


def reason_for(item: Item, profile: Profile, llm_call: LLMCall) -> str:
    return llm_call(SYSTEM_PROMPT, build_reason_prompt(item, profile))


def recommend(
    user_id: str,
    *,
    catalog: list[Item],
    profiles: dict[str, Profile],
    llm_call: LLMCall,
    k: int,
) -> dict:
    """Full pass: profile lookup -> deterministic rank -> LLM reason per item."""
    profile = profiles.get(user_id)
    if profile is None:
        raise KeyError(user_id)

    top = rank_items(catalog, profile, k)
    recommendations = []
    for s in top:
        recommendations.append(
            {
                "item_id": s.item.id,
                "title": s.item.title,
                "reason": reason_for(s.item, profile, llm_call),
            }
        )
    return {"recommendations": recommendations}
