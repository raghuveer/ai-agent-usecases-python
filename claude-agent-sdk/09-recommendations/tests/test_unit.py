# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC09 Personalised recommendations (claude-agent-sdk). See claude-agent-sdk/09-recommendations/README.md
"""Unit tests for UC09 recommendations (claude-agent-sdk). Stubbed agent, no network.

The load-bearing test is catalog grounding: a recommendation for a product that
does not exist must be rejected, not enriched-and-returned.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.recommend import (
    CATALOG_TOOL,
    EMIT_TOOL,
    PROFILE_TOOL,
    get_profile,
    list_catalog,
    recommend,
)
from app.settings import Settings

GOOD = {
    "items": [
        {"id": "p-2", "reason": "Ada likes office gear and it is within her $450 budget."},
        {"id": "p-6", "reason": "Pairs with the keyboard she just bought."},
    ],
    "rationale": "Focused on office productivity, skipping kitchen items she dislikes.",
}


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text="Recorded.",
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name=PROFILE_TOOL, input={"user_id": "u-1"}),
                ToolCall(name=CATALOG_TOOL, input={"category": "office"}),
                ToolCall(name=EMIT_TOOL, input=dict(GOOD)),
            ],
            num_turns=4,
            cost_usd=0.002,
        )

    return runner


def make_client(**kw) -> TestClient:
    return TestClient(create_app(runner=make_runner(**kw)))


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "claude-agent-sdk",
        "usecase": "09-recommendations",
    }


def test_run_returns_enriched_ranked_items():
    body = make_client().post("/run", json={"user_id": "u-1"}).json()
    assert body["valid"] is True
    assert [i["id"] for i in body["items"]] == ["p-2", "p-6"]
    # Enriched from the real catalog, not from the model's claim.
    assert body["items"][0]["name"] == "Standing Desk"
    assert body["items"][0]["price"] == 399.50
    assert body["rationale"].startswith("Focused on office")


def test_hallucinated_product_id_is_rejected():
    """The critical guard: an invented id must never reach the caller."""
    bad = {"items": [{"id": "p-999", "reason": "invented"}], "rationale": "x"}
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"user_id": "u-1"})
        .json()
    )
    assert body["valid"] is False
    assert body["items"] == []
    assert "p-999" in body["errors"][0]


def test_partially_hallucinated_list_is_rejected_wholesale():
    bad = {
        "items": [
            {"id": "p-2", "reason": "real"},
            {"id": "p-404", "reason": "invented"},
        ],
        "rationale": "x",
    }
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"user_id": "u-1"})
        .json()
    )
    assert body["valid"] is False
    assert "p-404" in body["errors"][0]


def test_missing_emit_is_reported():
    body = make_client(tool_calls=[]).post("/run", json={"user_id": "u-1"}).json()
    assert body["valid"] is False
    assert "did not call emit_recommendations" in body["errors"][0]


def test_empty_item_list_is_rejected():
    bad = {"items": [], "rationale": "nothing suits"}
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"user_id": "u-1"})
        .json()
    )
    assert body["valid"] is False


def test_too_many_items_rejected():
    bad = {
        "items": [{"id": "p-2", "reason": "r"} for _ in range(6)],
        "rationale": "x",
    }
    body = (
        make_client(tool_calls=[ToolCall(name=EMIT_TOOL, input=bad)])
        .post("/run", json={"user_id": "u-1"})
        .json()
    )
    assert body["valid"] is False


def test_catalog_and_profile_endpoints():
    client = make_client()
    assert len(client.get("/catalog").json()) == 7
    assert "u-1" in client.get("/profiles").json()


@pytest.mark.anyio
async def test_get_profile_returns_preferences():
    out = await get_profile.handler({"user_id": "U-1"})  # case-insensitive
    text = out["content"][0]["text"]
    assert "Ada" in text and "office" in text and "450" in text


@pytest.mark.anyio
async def test_get_profile_unknown_user_is_error():
    out = await get_profile.handler({"user_id": "nobody"})
    assert out["is_error"] is True


@pytest.mark.anyio
async def test_list_catalog_filters_by_category():
    out = await list_catalog.handler({"category": "kitchen"})
    text = out["content"][0]["text"]
    assert "Espresso Machine" in text and "Standing Desk" not in text


@pytest.mark.anyio
async def test_list_catalog_unknown_category_is_error():
    out = await list_catalog.handler({"category": "spacecraft"})
    assert out["is_error"] is True


@pytest.mark.anyio
async def test_list_catalog_without_filter_returns_everything():
    out = await list_catalog.handler({"category": ""})
    assert out["content"][0]["text"].count("\n") == 6  # 7 lines


@pytest.mark.anyio
async def test_options_register_all_three_tools():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        return AgentResult(text="", tool_calls=[ToolCall(name=EMIT_TOOL, input=dict(GOOD))])

    await recommend("u-1", Settings(), spy)
    assert seen["tools"] == [PROFILE_TOOL, CATALOG_TOOL, EMIT_TOOL]


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"user_id": "u-1"}


def _stub_runner(**fields):
    """A runner that returns exactly the AgentResult it is handed."""

    async def runner(prompt, options) -> AgentResult:
        return AgentResult(**fields)

    return runner


def test_stop_reason_reports_a_completed_run():
    """The SDK reports no stop reason on several paths, so the field would be
    null exactly when the run was fine. `end_turn` fills that gap: callers get
    one field that is always present and always means something."""
    client = TestClient(create_app(runner=_stub_runner(text="done", num_turns=1)))
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "end_turn"


def test_stop_reason_reports_a_capped_run():
    """The gap this closes. A run cut short by `max_turns` still answers 200
    with whatever it managed to produce -- previously indistinguishable from a
    run that finished properly, which is the one thing a caller must be able to
    tell apart."""
    client = TestClient(
        create_app(
            runner=_stub_runner(
                text="partial", num_turns=8, is_error=True, stop_reason="max_turns"
            )
        )
    )
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "max_turns"
