# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langchain). See langchain/07-multi-agent/README.md
"""Unit tests for UC7 multi-agent (langchain) — fully mocked, no network.

A ``RecordingFakeChat`` returns queued replies in order and records the prompt
each call received, so we can assert the orchestration ORDER (writer →
reviewer), that each sub-agent saw the upstream output, the aggregation shape,
and that the reject → revise path runs exactly once (capped).
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from app import agents, search
from app.main import app as fastapi_app


class RecordingFakeChat(FakeListChatModel):
    """FakeListChatModel that records the rendered prompt text of each call."""

    # pydantic model: declare the extra field so assignment is allowed.
    seen: list = []

    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)
        object.__setattr__(self, "seen", [])

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append("\n".join(m.content for m in messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


# --------------------------------------------------------------------------- #
# researcher: deterministic offline corpus search
# --------------------------------------------------------------------------- #
def test_research_finds_topic_facts():
    facts = search.research("vertical farming crops", top_k=4)
    assert facts
    assert "vertical" in " ".join(facts).lower() or "crops" in " ".join(facts).lower()


def test_research_no_match_returns_empty():
    assert search.research("zzz nonexistent qqq", top_k=4) == []
    assert "No relevant facts" in search.format_research([])


def test_parse_approved_redaction_tolerant():
    assert agents.parse_approved("Critique: fine\nAPPROVED: yes") is True
    assert agents.parse_approved("Critique: bad\nAPPROVED: no") is False
    assert agents.parse_approved("no verdict line here") is True


# --------------------------------------------------------------------------- #
# orchestration order + data flow
# --------------------------------------------------------------------------- #
def test_orchestration_order_and_dataflow_happy_path():
    llm = RecordingFakeChat([
        "A clear two sentence summary about the topic.",  # writer
        "Critique: faithful and clear.\nAPPROVED: yes",   # reviewer
    ])

    result = agents.orchestrate(
        "tidal energy", llm=llm, research_top_k=4, max_revisions=1
    )

    assert len(llm.seen) == 2
    writer_prompt, reviewer_prompt = llm.seen
    # Writer received the researcher's bullet facts.
    assert "Research notes:" in writer_prompt
    assert "tidal" in writer_prompt.lower()
    # Reviewer received the writer's draft.
    assert "A clear two sentence summary about the topic." in reviewer_prompt

    assert result.approved is True
    assert result.revisions == 0
    assert set(result.contributions) == {"research", "writer", "reviewer"}
    assert result.contributions["writer"] == result.draft
    assert result.contributions["reviewer"] == result.review
    assert "tidal" in result.contributions["research"].lower()


def test_reject_then_revise_runs_exactly_once():
    llm = RecordingFakeChat([
        "First draft.",
        "Critique: too vague.\nAPPROVED: no",
        "Second, improved draft.",
        "Critique: better now.\nAPPROVED: yes",
    ])

    result = agents.orchestrate(
        "vertical farming", llm=llm, research_top_k=4, max_revisions=1
    )

    assert len(llm.seen) == 4
    assert result.revisions == 1
    assert result.approved is True
    assert result.draft == "Second, improved draft."
    # The revise writer call carried the reviewer's critique.
    assert "too vague" in llm.seen[2]


def test_revise_loop_is_capped_at_max_revisions():
    llm = RecordingFakeChat([
        "draft 1", "Critique: no.\nAPPROVED: no",
        "draft 2", "Critique: still no.\nAPPROVED: no",
        "draft 3", "Critique: nope.\nAPPROVED: no",
    ])
    result = agents.orchestrate(
        "mesh networking", llm=llm, research_top_k=4, max_revisions=1
    )
    assert result.revisions == 1
    assert result.approved is False
    assert result.draft == "draft 2"
    assert len(llm.seen) == 4  # no third round


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel injected, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    fastapi_app.state.llm = FakeListChatModel(responses=[
        "A tidy summary of the topic.",
        "Critique: faithful.\nAPPROVED: yes",
    ])
    client = TestClient(fastapi_app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "07-multi-agent",
    }

    r = client.post("/run", json={"topic": "tidal energy"})
    assert r.status_code == 200
    body = r.json()
    assert body["draft"] == "A tidy summary of the topic."
    assert body["approved"] is True
    assert set(body["contributions"]) == {"research", "writer", "reviewer"}
    assert "tidal" in body["contributions"]["research"].lower()
