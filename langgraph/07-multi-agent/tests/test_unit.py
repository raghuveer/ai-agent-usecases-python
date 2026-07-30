# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (langgraph). See langgraph/07-multi-agent/README.md
"""Unit tests for UC7 multi-agent (langgraph) — fully mocked, no network.

A ``RecordingFakeChat`` scripts the writer/reviewer node replies and records the
prompt each node received, so the whole researcher → writer → reviewer →
(revise | END) graph runs offline. Tests assert orchestration ORDER, that each
sub-agent saw the upstream output (writer got the research, reviewer got the
draft), the aggregation shape, and that the reject → revise cycle runs once.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import graph, search
from app.main import create_app


class RecordingFakeChat(FakeListChatModel):
    """FakeListChatModel that records the rendered prompt text of each call."""

    seen: list = []

    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)
        object.__setattr__(self, "seen", [])

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append("\n".join(m.content for m in messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


# --------------------------------------------------------------------------- #
# researcher node logic: deterministic offline corpus search
# --------------------------------------------------------------------------- #
def test_research_finds_topic_facts():
    facts = search.research("mesh networking nodes", top_k=4)
    assert facts
    assert "mesh" in " ".join(facts).lower() or "node" in " ".join(facts).lower()


def test_research_no_match_returns_empty():
    assert search.research("zzz nonexistent qqq", top_k=4) == []
    assert "No relevant facts" in search.format_research([])


def test_parse_approved_redaction_tolerant():
    assert graph.parse_approved("Critique: fine\nAPPROVED: yes") is True
    assert graph.parse_approved("Critique: bad\nAPPROVED: no") is False
    assert graph.parse_approved("no verdict") is True


# --------------------------------------------------------------------------- #
# graph: order, shared-state dataflow, aggregation
# --------------------------------------------------------------------------- #
def test_order_and_dataflow_happy_path():
    llm = RecordingFakeChat([
        "A clear two sentence summary about the topic.",  # writer node
        "Critique: faithful and clear.\nAPPROVED: yes",   # reviewer node
    ])

    result = graph.run_multi_agent(
        "tidal energy", llm=llm, research_top_k=4, max_revisions=1
    )

    # Two LLM nodes ran in order: writer then reviewer (researcher is offline).
    assert len(llm.seen) == 2
    writer_prompt, reviewer_prompt = llm.seen
    # Writer read the researcher's facts from shared state.
    assert "Research notes:" in writer_prompt
    assert "tidal" in writer_prompt.lower()
    # Reviewer read the writer's draft from shared state.
    assert "A clear two sentence summary about the topic." in reviewer_prompt

    assert result["approved"] is True
    assert result["revisions"] == 0
    assert set(result["contributions"]) == {"research", "writer", "reviewer"}
    assert result["contributions"]["writer"] == result["draft"]
    assert result["contributions"]["reviewer"] == result["review"]
    assert "tidal" in result["contributions"]["research"].lower()


def test_reject_then_revise_runs_exactly_once():
    llm = RecordingFakeChat([
        "First draft.",
        "Critique: too vague.\nAPPROVED: no",
        "Second, improved draft.",
        "Critique: better now.\nAPPROVED: yes",
    ])

    result = graph.run_multi_agent(
        "vertical farming", llm=llm, research_top_k=4, max_revisions=1
    )

    assert len(llm.seen) == 4
    assert result["revisions"] == 1
    assert result["approved"] is True
    assert result["draft"] == "Second, improved draft."
    # The revise writer node carried the reviewer's critique through state.
    assert "too vague" in llm.seen[2]


def test_revise_loop_is_capped_at_max_revisions():
    llm = RecordingFakeChat([
        "draft 1", "Critique: no.\nAPPROVED: no",
        "draft 2", "Critique: still no.\nAPPROVED: no",
        "draft 3", "Critique: nope.\nAPPROVED: no",
    ])
    result = graph.run_multi_agent(
        "mesh networking", llm=llm, research_top_k=4, max_revisions=1
    )
    assert result["revisions"] == 1
    assert result["approved"] is False
    assert result["draft"] == "draft 2"
    assert len(llm.seen) == 4  # no third round


def test_graph_has_sub_agent_nodes_and_revise_edge():
    """The compiled graph exposes the sub-agent nodes + the revise cycle node."""
    llm = FakeListChatModel(responses=["Critique: ok\nAPPROVED: yes"])
    compiled = graph.build_multi_agent_graph(llm)
    nodes = set(compiled.get_graph().nodes.keys())
    assert {"researcher", "writer", "reviewer", "revise"} <= nodes


# --------------------------------------------------------------------------- #
# HTTP level (FakeListChatModel injected, no network)
# --------------------------------------------------------------------------- #
def make_client(responses: list[str]) -> TestClient:
    return TestClient(create_app(llm=FakeListChatModel(responses=responses)))


def test_health():
    client = make_client(["Critique: ok\nAPPROVED: yes"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "07-multi-agent",
    }


def test_run_returns_contract_shape():
    client = make_client([
        "A tidy summary of the topic.",
        "Critique: faithful.\nAPPROVED: yes",
    ])
    resp = client.post("/run", json={"topic": "tidal energy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["draft"] == "A tidy summary of the topic."
    assert body["approved"] is True
    assert set(body["contributions"]) == {"research", "writer", "reviewer"}
    assert "tidal" in body["contributions"]["research"].lower()


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
