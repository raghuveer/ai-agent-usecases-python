# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC9 Recommendations (raw-api). See raw-api/09-recommendations/README.md
"""Unit tests for UC9 recommendations (raw-api) — fully mocked, no network.

The ranking is deterministic plain Python, so we assert the exact top-k for a
fixture profile. The explanation LLM is a stub, so reasons are attached without
any network call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import llm, recommend
from app.main import create_app
from app.settings import Settings

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


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# --------------------------------------------------------------------------- #
# Deterministic ranking (no LLM)
# --------------------------------------------------------------------------- #
def test_load_bundled_data():
    catalog = recommend.load_catalog()
    profiles = recommend.load_profiles()
    assert len(catalog) >= 8
    assert "u1" in profiles
    assert all(it.title for it in catalog)


def test_score_item_counts_overlap():
    # m1: genres {sci-fi, adventure} both liked (2) + tags {space, epic} both liked (2) = 4
    assert recommend.score_item(CATALOG[0], PROFILE) == 4
    # m3: no overlap at all
    assert recommend.score_item(CATALOG[2], PROFILE) == 0


def test_rank_picks_correct_top_k():
    ranked = recommend.rank_items(CATALOG, PROFILE, k=2)
    ids = [s.item.id for s in ranked]
    # m1 (score 4) and m6 (sci-fi + space + epic = 3) beat m2 (sci-fi only = 1).
    assert ids == ["m1", "m6"]
    # Zero-overlap item m3 is dropped entirely.
    assert "m3" not in ids


def test_rank_drops_zero_overlap_and_caps_k():
    ranked = recommend.rank_items(CATALOG, PROFILE, k=10)
    ids = [s.item.id for s in ranked]
    assert ids == ["m1", "m6", "m2"]  # m3 dropped (0 overlap)


def test_recommend_attaches_reasons_with_stub_llm():
    calls = []

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        return "It matches your love of sci-fi space epics."

    result = recommend.recommend(
        "u1",
        catalog=CATALOG,
        profiles={"u1": PROFILE},
        llm_call=fake_llm,
        k=2,
    )
    recs = result["recommendations"]
    assert [r["item_id"] for r in recs] == ["m1", "m6"]
    assert all(r["reason"] for r in recs)
    # The LLM was asked once per recommended item and saw the profile grounding.
    assert len(calls) == 2
    assert "Ava" in calls[0][1]
    assert "sci-fi" in calls[0][1]


def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("qwen-local-instruct", "x") == "x"


def test_recommend_unknown_user_raises():
    import pytest

    with pytest.raises(KeyError):
        recommend.recommend(
            "nobody", catalog=CATALOG, profiles={"u1": PROFILE}, llm_call=lambda s, u: "x", k=3
        )


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client("A great fit for your tastes.")
    # Pre-set data so the startup hook does not need to load files (it would
    # anyway, but this keeps the test self-contained).
    app.state.catalog = CATALOG
    app.state.profiles = {"u1": PROFILE}

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "09-recommendations",
    }

    r = client.post("/run", json={"user_id": "u1", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert [rec["item_id"] for rec in body["recommendations"]] == ["m1", "m6"]
    assert all(rec["reason"] for rec in body["recommendations"])

    # Unknown user => 404.
    miss = client.post("/run", json={"user_id": "ghost"})
    assert miss.status_code == 404
