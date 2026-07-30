# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (langgraph). See langgraph/09-recommendations/README.md
"""Recommendations core for UC9 as a tiny langgraph StateGraph.

Pieces:
- Data loading (``load_catalog`` / ``load_profiles``) reads the bundled JSON.
- ``build_recommend_graph`` compiles a ``StateGraph`` with ``rank`` -> ``explain``
  -> END:
    * ``rank`` is deterministic plain Python — score each catalog item by overlap
      with the user's liked genres/tags, sort, take top-k. NO model involved.
    * ``explain`` calls the injected LLM once per ranked item for a grounded
      one-sentence reason.

The graph is intentionally tiny so the langgraph approach lines up apples-to-
apples with the raw-api and langchain versions. The retriever-equivalent (the
ranking) and the LLM are injected, so unit tests run fully offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import strip_thinking, system_prompt
from .settings import Settings, get_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_INSTRUCTION = (
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


def rank_items(catalog: list[Item], profile: Profile, k: int) -> list[Item]:
    """Rank by overlap score (desc), tie-break by item id (asc); take top-k.

    Items with zero overlap are dropped — we only recommend genuine matches.
    """
    scored = [(score_item(it, profile), it) for it in catalog]
    scored = [(s, it) for s, it in scored if s > 0]
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [it for _, it in scored[:k]]


def _reason_prompt(item: Item, profile: Profile) -> str:
    return (
        f"User {profile.name} likes genres {profile.liked_genres} and "
        f"tags {profile.liked_tags}.\n"
        f"Item '{item.title}' has genres {item.genres} and tags {item.tags}.\n"
        "Why does this item fit the user? Answer in one sentence."
    )


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class RecState(TypedDict):
    profile: Profile
    catalog: list[Item]
    k: int
    ranked: list[Item]
    recommendations: list[dict]


def build_recommend_graph(llm: BaseChatModel, settings: Settings | None = None):
    """Compile a rank -> explain -> END graph. The LLM is injected."""
    settings = settings or get_settings()

    def rank(state: RecState) -> dict:
        ranked = rank_items(state["catalog"], state["profile"], state["k"])
        return {"ranked": ranked}

    def explain(state: RecState) -> dict:
        profile = state["profile"]
        recommendations = []
        for item in state["ranked"]:
            messages = [
                SystemMessage(content=system_prompt(SYSTEM_INSTRUCTION, settings)),
                HumanMessage(content=_reason_prompt(item, profile)),
            ]
            reason = strip_thinking(llm.invoke(messages).content)
            recommendations.append(
                {"item_id": item.id, "title": item.title, "reason": reason}
            )
        return {"recommendations": recommendations}

    graph = StateGraph(RecState)
    graph.add_node("rank", rank)
    graph.add_node("explain", explain)
    graph.set_entry_point("rank")
    graph.add_edge("rank", "explain")
    graph.add_edge("explain", END)
    return graph.compile()
