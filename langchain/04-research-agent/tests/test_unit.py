# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (langchain). See langchain/04-research-agent/README.md
"""Unit tests for UC4 research-agent (langchain) — fully mocked, no network."""
from __future__ import annotations

from langchain_community.chat_models.fake import FakeListChatModel
from fastapi.testclient import TestClient

from app import agent, llm
from app.main import create_app


# --------------------------------------------------------------------------- #
# Corpus + search tool (deterministic, offline)
# --------------------------------------------------------------------------- #
def test_load_corpus_reads_bundled_notes():
    sources = {p.source for p in agent.load_corpus()}
    assert {
        "company.md",
        "products.md",
        "returns.md",
        "warranty.md",
        "support.md",
    } <= sources


def test_search_is_deterministic():
    corpus = agent.Corpus()
    a = corpus.search("refund original packaging restocking", top_k=3)
    b = corpus.search("refund original packaging restocking", top_k=3)
    assert [p.source for p in a] == [p.source for p in b]
    assert a[0].source == "returns.md"


def test_search_tool_records_sources():
    corpus = agent.Corpus()
    tool = agent.make_search_tool(corpus, top_k=2)
    assert tool.name == "search"
    obs = tool.func("warranty defects coverage")
    assert "[warranty.md]" in obs
    assert "warranty.md" in corpus.last_sources


def test_search_no_match():
    corpus = agent.Corpus()
    obs = corpus.search_text("xyzzy nonexistent token", top_k=3)
    assert "No matching" in obs
    assert corpus.last_sources == []


# --------------------------------------------------------------------------- #
# ReAct parsing
# --------------------------------------------------------------------------- #
def test_parse_action_and_final():
    a = agent.parse_step("Thought: t\nAction: search\nAction Input: return window")
    assert a.action == "search" and a.action_input == "return window"
    f = agent.parse_step("Thought: done\nFinal Answer: 30 days (returns.md).")
    assert f.final_answer == "30 days (returns.md)."


def test_parse_tolerates_redacted_label():
    p = agent.parse_step("Action: search\n<PERSON>: return window warranty")
    assert p.action == "search"
    assert p.action_input == "return window warranty"


# --------------------------------------------------------------------------- #
# The ReAct loop with a scripted FakeListChatModel
# --------------------------------------------------------------------------- #
def test_loop_runs_search_then_finishes():
    fake = FakeListChatModel(
        responses=[
            "Thought: find returns.\nAction: search\nAction Input: return window",
            "Thought: done.\nFinal Answer: The return window is 30 days (returns.md).",
        ]
    )
    captured: list[str] = []

    class SpyCorpus(agent.Corpus):
        def search_text(self, query, top_k=3):
            captured.append(query)
            self.last_sources = ["returns.md"]
            return "[returns.md] 30-day return window"

    result = agent.run_agent(
        "What is the return window?", corpus=SpyCorpus(), llm=fake, max_steps=6
    )

    assert captured == ["return window"]  # tool got the PARSED input
    assert len(result.steps) == 1
    assert result.steps[0].action == "search"
    assert "30-day return window" in result.steps[0].observation
    assert result.sources == ["returns.md"]
    assert result.stopped_reason == "final_answer"
    assert "30 days" in result.answer


def test_loop_caps_steps():
    fake = FakeListChatModel(
        responses=["Thought: again\nAction: search\nAction Input: anything"] * 10
    )
    result = agent.run_agent("endless?", corpus=agent.Corpus(), llm=fake, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


def test_loop_dedupes_sources():
    fake = FakeListChatModel(
        responses=[
            "Action: search\nAction Input: a",
            "Action: search\nAction Input: b",
            "Final Answer: done (returns.md)",
        ]
    )

    class OneSource(agent.Corpus):
        def search_text(self, query, top_k=3):
            self.last_sources = ["returns.md"]
            return "[returns.md] x"

    result = agent.run_agent("q", corpus=OneSource(), llm=fake, max_steps=6)
    assert result.sources == ["returns.md"]


# --------------------------------------------------------------------------- #
# HTTP level (injected fake corpus + FakeListChatModel, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    fake = FakeListChatModel(
        responses=[
            "Thought: returns.\nAction: search\nAction Input: refund original packaging restocking",
            "Thought: warranty.\nAction: search\nAction Input: warranty defects coverage",
            "Thought: done.\nFinal Answer: 30-day returns; 1-year warranty "
            "(returns.md, warranty.md).",
        ]
    )
    app = create_app(corpus=agent.Corpus(), llm=fake)
    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "04-research-agent",
    }

    r = client.post("/run", json={"question": "returns and warranty?"})
    assert r.status_code == 200
    body = r.json()
    assert "warranty" in body["answer"].lower()
    assert "returns.md" in body["sources"]
    assert "warranty.md" in body["sources"]
    assert body["steps"][0]["action"] == "search"


# --------------------------------------------------------------------------- #
# Stop-sequence handling (the gateway's Anthropic path 500s on `stop`)
# --------------------------------------------------------------------------- #
def test_supports_stop_is_false_only_for_claude_models():
    assert llm.model_profile("qwen-local-instruct")["supports_stop"] is True
    assert llm.model_profile("qwen3:1.7b")["supports_stop"] is True
    assert llm.model_profile("claude-haiku")["supports_stop"] is False


def test_stop_sequences_omitted_for_claude_models():
    """Regression: ChatOpenAI must not be built with `stop` for claude-*."""
    assert llm.stop_sequences("claude-haiku") is None
    assert llm.stop_sequences("qwen-local-instruct") == list(llm.STOP_MARKERS)


def test_truncate_at_stop_cuts_hallucinated_observation():
    text = "Thought: look it up\nAction: search\nObservation: fabricated!"
    assert llm.truncate_at_stop(text) == "Thought: look it up\nAction: search"
    assert llm.truncate_at_stop("Thought: done  ") == "Thought: done"


def test_loop_ignores_a_model_supplied_observation():
    """With `stop` unsupported the model may write its own Observation; the loop
    must cut it rather than treat the fabrication as a real tool result."""
    fake = FakeListChatModel(
        responses=[
            "Thought: check\nAction: search\nAction Input: returns\n"
            "Observation: the window is 999 days\nFinal Answer: 999 days",
            "Final Answer: 30 days",
        ]
    )
    result = agent.run_agent("How long?", corpus=agent.Corpus(), llm=fake, max_steps=3)
    assert "999" not in result.answer
