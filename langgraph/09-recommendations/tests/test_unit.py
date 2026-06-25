# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (langgraph). See langgraph/09-recommendations/README.md
"""Unit tests for UC9 recommendations (langgraph). Fake LLM, no network.

Ranking is deterministic plain Python, so we assert the exact top-k for a fixture
profile. The explain node uses ``FakeListChatModel``, so reasons are attached
without any network.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import recommend
from app.main import create_app

CATALOG = [
    recommend.Item("m1", "Starfall Horizon", ["sci-fi", "adventure"], ["space", "epic"]),
    recommend.Item("m2", "The Last Algorithm", ["sci-fi", "thriller"], ["ai", "suspense"]),
    recommend.Item("m3", "Quiet Harbor", ["drama", "romance"], ["slow-burn"]),
    recommend.Item("m6", "Nebula Heist", ["sci-fi", "action"], ["space", "heist", "epic"]),
]

PROFILE = recommend.Profile(
    user_id="u1",
    name="Ava",
    liked_genres=["sci-fi", "adventure"],
    liked_tags=["space", "epic"],
)


def _client(responses: list[str]) -> TestClient:
    llm = FakeListChatModel(responses=responses)
    return TestClient(create_app(catalog=CATALOG, profiles={"u1": PROFILE}, llm=llm))


# --------------------------------------------------------------------------- #
# Deterministic ranking (no LLM)
# --------------------------------------------------------------------------- #
def test_load_bundled_data():
    catalog = recommend.load_catalog()
    profiles = recommend.load_profiles()
    assert len(catalog) >= 8
    assert "u1" in profiles


def test_score_item_counts_overlap():
    assert recommend.score_item(CATALOG[0], PROFILE) == 4  # m1
    assert recommend.score_item(CATALOG[2], PROFILE) == 0  # m3


def test_rank_picks_correct_top_k():
    ranked = recommend.rank_items(CATALOG, PROFILE, k=2)
    assert [it.id for it in ranked] == ["m1", "m6"]


def test_rank_drops_zero_overlap():
    ranked = recommend.rank_items(CATALOG, PROFILE, k=10)
    assert [it.id for it in ranked] == ["m1", "m6", "m2"]  # m3 dropped


def test_graph_ranks_then_explains():
    """The compiled graph runs rank then explain, attaching a reason per item."""
    llm = FakeListChatModel(responses=["reason A", "reason B"])
    graph = recommend.build_recommend_graph(llm)
    out = graph.invoke(
        {
            "profile": PROFILE,
            "catalog": CATALOG,
            "k": 2,
            "ranked": [],
            "recommendations": [],
        }
    )
    assert [it.id for it in out["ranked"]] == ["m1", "m6"]
    recs = out["recommendations"]
    assert [r["item_id"] for r in recs] == ["m1", "m6"]
    assert all(r["reason"] for r in recs)


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel, no network)
# --------------------------------------------------------------------------- #
def test_health():
    client = _client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "09-recommendations",
    }


def test_run_returns_ranked_recommendations():
    client = _client(["because space epic", "because sci-fi heist"])
    resp = client.post("/run", json={"user_id": "u1", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert [r["item_id"] for r in body["recommendations"]] == ["m1", "m6"]
    assert all(r["reason"] for r in body["recommendations"])


def test_run_unknown_user_404():
    client = _client(["unused"])
    resp = client.post("/run", json={"user_id": "ghost"})
    assert resp.status_code == 404
