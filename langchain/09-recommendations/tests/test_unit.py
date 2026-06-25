"""Unit tests for UC9 recommendations (langchain) — fully mocked, no network.

Ranking is deterministic plain Python, so we assert the exact top-k for a fixture
profile. The explanation LLM is ``FakeListChatModel``, so reasons are attached via
the LCEL chain without any network.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import recommend
from app.main import app

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


def make_client(responses: list[str]) -> TestClient:
    app.state.catalog = CATALOG
    app.state.profiles = {"u1": PROFILE}
    app.state.llm = FakeListChatModel(responses=responses)
    return TestClient(app)


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
    assert [s.item.id for s in ranked] == ["m1", "m6"]


def test_rank_drops_zero_overlap():
    ranked = recommend.rank_items(CATALOG, PROFILE, k=10)
    assert [s.item.id for s in ranked] == ["m1", "m6", "m2"]  # m3 dropped


def test_reason_chain_uses_profile_and_item():
    llm = FakeListChatModel(responses=["A grounded reason."])
    chain = recommend.build_reason_chain(llm, "qwen-local-instruct")
    out = chain.invoke(
        {
            "name": "Ava",
            "liked_genres": ["sci-fi"],
            "liked_tags": ["space"],
            "title": "Starfall Horizon",
            "genres": ["sci-fi"],
            "tags": ["space"],
        }
    )
    assert out == "A grounded reason."


def test_recommend_attaches_reasons():
    llm = FakeListChatModel(responses=["reason A", "reason B"])
    result = recommend.recommend(
        "u1", CATALOG, {"u1": PROFILE}, llm, "qwen-local-instruct", k=2
    )
    recs = result["recommendations"]
    assert [r["item_id"] for r in recs] == ["m1", "m6"]
    assert all(r["reason"] for r in recs)


def test_no_think_prefix_for_qwen3():
    assert recommend._system_prompt("qwen3:1.7b").startswith("/no_think")
    assert not recommend._system_prompt("qwen-local-instruct").startswith("/no_think")


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel, no network)
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "09-recommendations",
    }


def test_run_returns_ranked_recommendations():
    client = make_client(["because space epic", "because sci-fi heist"])
    resp = client.post("/run", json={"user_id": "u1", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert [r["item_id"] for r in body["recommendations"]] == ["m1", "m6"]
    assert all(r["reason"] for r in body["recommendations"])


def test_run_unknown_user_404():
    client = make_client(["unused"])
    resp = client.post("/run", json={"user_id": "ghost"})
    assert resp.status_code == 404
