# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 Human-in-the-loop approval (langgraph). See langgraph/10-hitl-approval/README.md
"""Unit tests for UC10 hitl-approval (langgraph). Fake LLM, no network.

Verifies the interrupt/resume mechanics: /run drafts then pauses (no execute),
/resume approved -> executed, /resume not-approved -> rejected, unknown/terminal
run_id -> 404.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.hitl import build_approval_graph
from app.main import create_app

DRAFT = "Your refund of $40 has been approved and will post in 3-5 days."


def make_client(draft: str = DRAFT) -> TestClient:
    # One response per /run (graph calls the LLM exactly once, in `draft`).
    llm = FakeListChatModel(responses=[draft] * 10)
    return TestClient(create_app(llm=llm))


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "10-hitl-approval",
    }


def test_run_pauses_with_proposed_action():
    client = make_client()
    r = client.post("/run", json={"request": "approve a $40 refund"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_approval"
    assert body["proposed_action"] == DRAFT
    assert body["run_id"]


def test_run_does_not_execute_before_resume():
    """The graph must SUSPEND at the interrupt — execute has not run yet."""
    llm = FakeListChatModel(responses=[DRAFT] * 10)
    app = create_app(llm=llm)
    client = TestClient(app)
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    # The checkpointed state is paused with `review`/`execute` still pending.
    snapshot = app.state.graph.get_state({"configurable": {"thread_id": run_id}})
    assert snapshot.next  # non-empty => graph is paused, not finished
    assert snapshot.values["status"] == "awaiting_approval"


def test_resume_approved_executes():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    r = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "executed"
    assert DRAFT in body["result"]
    assert "EXECUTED" in body["result"]


def test_resume_not_approved_rejects():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    r = client.post(
        "/resume", json={"run_id": run_id, "approved": False, "feedback": "too much"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rejected"
    assert body["result"] is None


def test_resume_unknown_run_id_404():
    client = make_client()
    r = client.post("/resume", json={"run_id": "nope", "approved": True})
    assert r.status_code == 404


def test_resume_twice_is_404_after_terminal():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    first = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert first.status_code == 200
    # State is terminal (no pending nodes) => second resume is unknown => 404.
    second = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert second.status_code == 404


def test_graph_suspends_at_interrupt_directly():
    """Drive the compiled graph directly: it interrupts before `execute`."""
    llm = FakeListChatModel(responses=[DRAFT])
    graph = build_approval_graph(llm)
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(
        {
            "request": "refund",
            "proposed_action": "",
            "approved": None,
            "feedback": None,
            "status": "",
            "result": None,
        },
        config=config,
    )
    snapshot = graph.get_state(config)
    # Paused before `execute`: `review` is still the pending task and carries
    # the interrupt payload with the proposed action.
    assert snapshot.next == ("review",)
    payloads = [
        itr.value
        for task in snapshot.tasks
        for itr in getattr(task, "interrupts", ())
    ]
    assert payloads and payloads[0]["proposed_action"] == DRAFT


def test_strip_thinking_removes_qwen3_think_blocks():
    """`/no_think` still emits an empty <think></think> pair; it must not ship.

    Found by running the Docker quickstart, whose default model is a qwen3 tag:
    answers came back with a leading empty thinking block before the text.
    """
    from app.llm import strip_thinking

    empty_block = "<think>" + "\n\n" + "</think>" + "\n\n" + "30 days."
    assert strip_thinking(empty_block) == "30 days."
    assert strip_thinking("<think>reasoning</think>Answer.") == "Answer."
    assert strip_thinking("  30 days.  ") == "30 days."
    # Chain-of-thought must never survive into a response.
    assert "reasoning" not in strip_thinking("<think>reasoning</think>ok")
