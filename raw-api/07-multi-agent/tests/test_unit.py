# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent orchestration (raw-api). See raw-api/07-multi-agent/README.md
"""Unit tests for UC7 multi-agent (raw-api) — fully mocked, no network.

A scripted ``llm_call`` returns queued replies per (system, user) call so the
orchestrator runs offline and deterministically. Tests assert:
- the deterministic researcher finds the right corpus facts (and the empty case),
- orchestration ORDER is researcher → writer → reviewer,
- each sub-agent received the upstream output (writer got the research; reviewer
  got the draft),
- the aggregation shape is correct,
- the reject → revise path runs exactly once (capped).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import agents, search
from app.agents import WRITER_SYSTEM, REVIEWER_SYSTEM
from app.main import create_app
from app.settings import Settings


class ScriptedLLM:
    """Returns queued replies in order, recording (system, user) it was sent.

    Roles are distinguished by the system prompt so tests can assert which
    sub-agent saw which upstream output.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._replies.pop(0) if self._replies else "APPROVED: yes"


# --------------------------------------------------------------------------- #
# researcher: deterministic offline corpus search
# --------------------------------------------------------------------------- #
def test_research_finds_topic_facts():
    facts = search.research("tidal energy ocean", top_k=4)
    assert facts, "expected matches for a corpus topic"
    joined = " ".join(facts).lower()
    assert "tidal" in joined or "tide" in joined


def test_research_no_match_returns_empty():
    assert search.research("zzz nonexistent qqq", top_k=4) == []
    assert "No relevant facts" in search.format_research([])


def test_parse_approved_redaction_tolerant():
    assert agents.parse_approved("Critique: fine\nAPPROVED: yes") is True
    assert agents.parse_approved("Critique: bad\nAPPROVED: no") is False
    # Missing token defaults to approved (the revise loop is the real safety net).
    assert agents.parse_approved("Critique: unclear verdict") is True


# --------------------------------------------------------------------------- #
# orchestration order + data flow between sub-agents
# --------------------------------------------------------------------------- #
def test_orchestration_order_and_dataflow_happy_path():
    script = ScriptedLLM([
        "A clear two sentence summary about the topic.",   # writer draft
        "Critique: faithful and clear.\nAPPROVED: yes",    # reviewer approves
    ])

    result = agents.orchestrate(
        "tidal energy", llm_call=script, research_top_k=4, max_revisions=1
    )

    # Exactly two LLM calls: writer then reviewer (researcher is deterministic).
    assert len(script.calls) == 2
    (sys1, user1), (sys2, user2) = script.calls
    assert sys1 == WRITER_SYSTEM          # writer ran first
    assert sys2 == REVIEWER_SYSTEM        # reviewer ran second

    # Writer received the researcher's bullet facts.
    assert "Research notes:" in user1
    assert "tidal" in user1.lower()
    # Reviewer received the writer's draft.
    assert "A clear two sentence summary about the topic." in user2

    # Aggregation shape.
    assert result.approved is True
    assert result.revisions == 0
    assert set(result.contributions) == {"research", "writer", "reviewer"}
    assert result.contributions["writer"] == result.draft
    assert result.contributions["reviewer"] == result.review
    assert "tidal" in result.contributions["research"].lower()


def test_reject_then_revise_runs_exactly_once():
    script = ScriptedLLM([
        "First draft.",                                   # writer draft 1
        "Critique: too vague.\nAPPROVED: no",             # reviewer rejects
        "Second, improved draft.",                        # writer draft 2 (revise)
        "Critique: better now.\nAPPROVED: yes",           # reviewer approves
    ])

    result = agents.orchestrate(
        "vertical farming", llm_call=script, research_top_k=4, max_revisions=1
    )

    # writer, reviewer, writer(revise), reviewer = 4 calls.
    assert len(script.calls) == 4
    assert result.revisions == 1
    assert result.approved is True
    assert result.draft == "Second, improved draft."
    # The revise writer call carried the reviewer's critique.
    revise_user = script.calls[2][1]
    assert "too vague" in revise_user


def test_revise_loop_is_capped_at_max_revisions():
    # Reviewer always rejects; the loop must stop after max_revisions and return
    # the last (still-unapproved) draft.
    script = ScriptedLLM([
        "draft 1", "Critique: no.\nAPPROVED: no",
        "draft 2", "Critique: still no.\nAPPROVED: no",
        "draft 3", "Critique: nope.\nAPPROVED: no",
    ])
    result = agents.orchestrate(
        "mesh networking", llm_call=script, research_top_k=4, max_revisions=1
    )
    assert result.revisions == 1
    assert result.approved is False
    assert result.draft == "draft 2"
    # 2 rounds × (writer + reviewer) = 4 calls; no third round.
    assert len(script.calls) == 4


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def _fake_openai_client(replies: list[str]) -> MagicMock:
    client = MagicMock()
    queue = list(replies)

    def _create(**kwargs):
        text = queue.pop(0) if queue else "Critique: ok\nAPPROVED: yes"
        if kwargs.get("stream"):
            # The streaming path consumes deltas, not a message. Split into a
            # few chunks so the tests exercise reassembly rather than one blob.
            def chunks():
                for piece in [text[i:i + 8] for i in range(0, len(text), 8)]:
                    delta = MagicMock()
                    delta.content = piece
                    choice = MagicMock()
                    choice.delta = delta
                    yield MagicMock(choices=[choice])

            return chunks()
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        return MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=100, completion_tokens=20),
        )

    client.chat.completions.create.side_effect = _create
    return client


def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client([
        "A tidy summary of the topic.",
        "Critique: faithful.\nAPPROVED: yes",
    ])

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "07-multi-agent",
    }

    r = client.post("/run", json={"topic": "tidal energy"})
    assert r.status_code == 200
    body = r.json()
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


# --------------------------------------------------------------------------- #
# Tracing + streaming. See docs/trace-format.md and docs/streaming.md
# --------------------------------------------------------------------------- #
def _client_with(replies):
    from app.main import create_app
    from app.settings import Settings

    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(list(replies))
    return TestClient(app)


_TEAM_SCRIPT = ["A short draft.", "Critique: fine.\nAPPROVED: yes"]


def test_trace_absent_unless_requested():
    body = _client_with(_TEAM_SCRIPT).post("/run", json={"topic": "x"}).json()
    assert body["trace"] is None


def test_trace_records_one_span_per_role_call():
    trace = _client_with(_TEAM_SCRIPT).post(
        "/run?trace=1", json={"topic": "x"}
    ).json()["trace"]

    assert trace["approach"] == "raw-api"
    assert trace["usecase"] == "07-multi-agent"
    # The researcher is deterministic (no model call); writer + reviewer are not.
    assert [s["type"] for s in trace["spans"]] == ["llm", "llm"]
    assert trace["outcome"]["stop_reason"] == "approved"


def test_stream_reports_the_handoffs():
    """`role` frames are what UC07 adds: the orchestration order, live."""
    import json as _json

    with _client_with(_TEAM_SCRIPT).stream(
        "POST", "/run/stream", json={"topic": "x"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    # Parse event/data pairs properly: `artifact` frames carry a "name" too, so
    # filtering on the field alone mixes the two kinds.
    roles, current = [], None
    for line in body.splitlines():
        if line.startswith("event: "):
            current = line.split(": ", 1)[1]
        elif line.startswith("data: ") and current == "role":
            roles.append(_json.loads(line[6:])["name"])
    assert roles[:3] == ["researcher", "writer", "reviewer"]
    assert "event: artifact" in body
    assert "event: final" in body
    assert body.endswith("\n\n")


def test_blocking_and_streaming_agree():
    blocking = _client_with(_TEAM_SCRIPT).post("/run", json={"topic": "x"}).json()
    with _client_with(_TEAM_SCRIPT).stream(
        "POST", "/run/stream", json={"topic": "x"}
    ) as resp:
        body = "".join(resp.iter_text())
    import json as _json

    final = _json.loads(body.rsplit("data: ", 1)[1].strip())
    assert blocking["draft"] == final["draft"]
    assert blocking["approved"] == final["approved"]
