"""Unit tests for UC10 hitl-approval (langchain). Fake LLM, no network."""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import hitl
from app.main import create_app

DRAFT = "Your refund of $40 has been approved and will post in 3-5 days."


def make_client(draft: str = DRAFT) -> TestClient:
    llm = FakeListChatModel(responses=[draft] * 10)
    return TestClient(create_app(llm=llm))


def _draft_chain(draft: str = DRAFT):
    return hitl.build_draft_chain(FakeListChatModel(responses=[draft] * 10))


# --------------------------------------------------------------------------- #
# Core (chain + registry), no HTTP.
# --------------------------------------------------------------------------- #
def test_start_run_drafts_and_pauses():
    registry = hitl.RunRegistry()
    run = hitl.start_run("approve a $40 refund", chain=_draft_chain(), registry=registry)
    assert run.status == "awaiting_approval"
    assert run.proposed_action == DRAFT
    # The paused run is persisted (manual pause workaround).
    assert registry.get(run.run_id) is run


def test_resume_approved_executes_and_locks():
    registry = hitl.RunRegistry()
    run = hitl.start_run("refund", chain=_draft_chain(), registry=registry)
    out = hitl.resume_run(run.run_id, approved=True, registry=registry)
    assert out["status"] == "executed"
    assert DRAFT in out["result"]
    assert "EXECUTED" in out["result"]
    # Gone after a terminal resume — cannot resume twice.
    assert registry.get(run.run_id) is None


def test_resume_not_approved_rejects():
    registry = hitl.RunRegistry()
    run = hitl.start_run("refund", chain=_draft_chain(), registry=registry)
    out = hitl.resume_run(
        run.run_id, approved=False, feedback="too high", registry=registry
    )
    assert out["status"] == "rejected"
    assert out["result"] is None
    assert out["feedback"] == "too high"
    assert registry.get(run.run_id) is None


def test_resume_unknown_run_raises():
    registry = hitl.RunRegistry()
    try:
        hitl.resume_run("nope", approved=True, registry=registry)
        assert False, "expected UnknownRunError"
    except hitl.UnknownRunError:
        pass


# --------------------------------------------------------------------------- #
# HTTP level (fake LLM, no network).
# --------------------------------------------------------------------------- #
def test_health():
    h = make_client().get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "langchain",
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


def test_run_then_resume_approved_executes():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    r = client.post("/resume", json={"run_id": run_id, "approved": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "executed"
    assert DRAFT in body["result"]


def test_run_then_resume_rejected():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    r = client.post(
        "/resume", json={"run_id": run_id, "approved": False, "feedback": "no"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_resume_unknown_run_id_404():
    client = make_client()
    r = client.post("/resume", json={"run_id": "nope", "approved": True})
    assert r.status_code == 404


def test_resume_twice_is_404_after_terminal():
    client = make_client()
    run_id = client.post("/run", json={"request": "approve refund"}).json()["run_id"]
    assert client.post("/resume", json={"run_id": run_id, "approved": True}).status_code == 200
    assert client.post("/resume", json={"run_id": run_id, "approved": True}).status_code == 404
