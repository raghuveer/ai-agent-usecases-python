# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (raw-api). See raw-api/10-hitl-approval/README.md
"""Unit tests for UC10 hitl-approval (raw-api) — fully mocked, no network."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import hitl, llm
from app.main import create_app
from app.settings import Settings


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def _client(reply: str = "Your refund of $40 has been approved.") -> TestClient:
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(reply)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Core (store + draft/resume), no HTTP.
# --------------------------------------------------------------------------- #
def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("qwen-local-instruct", "x") == "x"


def test_start_run_drafts_and_persists_paused_run():
    store = hitl.CheckpointStore()
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Approved: refund of $40 issued to your card."

    run = hitl.start_run("Approve a $40 refund", llm_call=fake_llm, store=store)

    assert run.status == "awaiting_approval"
    assert run.proposed_action == "Approved: refund of $40 issued to your card."
    # The request reached the draft prompt and the run was persisted.
    assert "Approve a $40 refund" in captured["prompt"]
    assert store.get(run.run_id) is run


def test_resume_approved_executes_and_locks_state():
    store = hitl.CheckpointStore()
    run = hitl.start_run("refund", llm_call=lambda p: "Refund approved.", store=store)

    out = hitl.resume_run(run.run_id, approved=True, store=store)
    assert out["status"] == "executed"
    assert "Refund approved." in out["result"]
    assert "EXECUTED" in out["result"]
    # State is gone after a terminal resume — cannot resume twice.
    assert store.get(run.run_id) is None


def test_resume_not_approved_rejects():
    store = hitl.CheckpointStore()
    run = hitl.start_run("refund", llm_call=lambda p: "Refund approved.", store=store)

    out = hitl.resume_run(
        run.run_id, approved=False, feedback="amount too high", store=store
    )
    assert out["status"] == "rejected"
    assert out["result"] is None
    assert out["feedback"] == "amount too high"
    assert store.get(run.run_id) is None


def test_resume_unknown_run_raises():
    store = hitl.CheckpointStore()
    try:
        hitl.resume_run("does-not-exist", approved=True, store=store)
        assert False, "expected UnknownRunError"
    except hitl.UnknownRunError:
        pass


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network).
# --------------------------------------------------------------------------- #
def test_health():
    client = _client()
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "10-hitl-approval",
    }


def test_run_pauses_with_proposed_action():
    client = _client("Your refund of $40 has been approved.")
    r = client.post("/run", json={"request": "approve a $40 refund"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_approval"
    assert body["proposed_action"] == "Your refund of $40 has been approved."
    assert body["run_id"]


def test_run_then_resume_approved_executes():
    client = _client("Your refund of $40 has been approved.")
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]

    r = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "executed"
    assert "approved" in body["result"].lower()


def test_run_then_resume_rejected():
    client = _client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]

    r = client.post(
        "/resume",
        json={"run_id": run_id, "approved": False, "feedback": "too risky"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_resume_unknown_run_id_404():
    client = _client()
    r = client.post("/resume", json={"run_id": "nope", "approved": True})
    assert r.status_code == 404


def test_resume_twice_is_404_after_terminal():
    client = _client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    first = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert first.status_code == 200
    second = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert second.status_code == 404
