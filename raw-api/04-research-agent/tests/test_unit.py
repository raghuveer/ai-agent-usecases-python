# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC4 Research agent (raw-api). See raw-api/04-research-agent/README.md
"""Unit tests for UC4 research-agent (raw-api) — fully mocked, no network."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import agent, llm
from app.main import create_app
from app.settings import Settings


# --------------------------------------------------------------------------- #
# Corpus + search tool (deterministic, offline)
# --------------------------------------------------------------------------- #
def test_load_corpus_reads_bundled_notes():
    passages = agent.load_corpus()
    sources = {p.source for p in passages}
    assert {
        "company.md",
        "products.md",
        "returns.md",
        "warranty.md",
        "support.md",
    } <= sources
    assert all(p.text for p in passages)


def test_search_is_deterministic_and_scored():
    corpus = agent.Corpus()
    hits1 = corpus.search("refund original packaging restocking", top_k=3)
    hits2 = corpus.search("refund original packaging restocking", top_k=3)
    assert [h.source for h in hits1] == [h.source for h in hits2]  # deterministic
    assert hits1, "expected matches for a return query"
    # The returns note is the strongest match for refund/packaging/restocking.
    assert hits1[0].source == "returns.md"
    # A warranty query surfaces the warranty note.
    assert "warranty.md" in {h.source for h in corpus.search("warranty defects", top_k=3)}


def test_search_text_collects_sources():
    corpus = agent.Corpus()
    obs, sources = corpus.search_text("warranty coverage defects", top_k=2)
    assert "warranty.md" in sources
    assert "[warranty.md]" in obs


def test_search_no_match_returns_empty():
    corpus = agent.Corpus()
    obs, sources = corpus.search_text("xyzzy nonexistent token", top_k=3)
    assert sources == []
    assert "No matching" in obs


# --------------------------------------------------------------------------- #
# ReAct parsing
# --------------------------------------------------------------------------- #
def test_parse_action_step():
    p = agent.parse_step(
        "Thought: I should look it up\nAction: search\nAction Input: return window"
    )
    assert p.final_answer is None
    assert p.action == "search"
    assert p.action_input == "return window"
    assert "look it up" in p.thought


def test_parse_final_answer_wins():
    p = agent.parse_step(
        "Thought: done\nFinal Answer: The return window is 30 days (returns.md)."
    )
    assert p.final_answer == "The return window is 30 days (returns.md)."


def test_parse_defensive_no_fields():
    p = agent.parse_step("just some prose with no protocol")
    assert p.action == ""
    assert p.final_answer is None


def test_parse_tolerates_redacted_action_input_label():
    """The gateway PII filter can rewrite the 'Action Input' label to a token
    like '<PERSON>:'. We still recover the action + query."""
    p = agent.parse_step(
        "Thought: search it\nAction: search\n<PERSON>: return window warranty"
    )
    assert p.action == "search"
    assert p.action_input == "return window warranty"
    assert p.final_answer is None


# --------------------------------------------------------------------------- #
# The ReAct loop with a scripted (mocked) LLM
# --------------------------------------------------------------------------- #
def test_loop_runs_search_then_finishes():
    """Scripted: one search action, then a Final Answer."""
    scripted = iter(
        [
            "Thought: I need the return policy.\nAction: search\nAction Input: return window",
            "Thought: I have it.\nFinal Answer: The return window is 30 days (returns.md).",
        ]
    )
    captured_inputs: list[str] = []

    class SpyCorpus(agent.Corpus):
        def search_text(self, query, top_k=3):
            captured_inputs.append(query)
            return ("[returns.md] 30-day return window", ["returns.md"])

    def fake_llm(_user_prompt: str) -> str:
        return next(scripted)

    result = agent.run_agent(
        "What is the return window?",
        corpus=SpyCorpus(),
        llm_call=fake_llm,
        max_steps=6,
    )

    # The search tool was invoked with the PARSED input.
    assert captured_inputs == ["return window"]
    # One step recorded with the observation appended.
    assert len(result.steps) == 1
    assert result.steps[0].action == "search"
    assert result.steps[0].action_input == "return window"
    assert "30-day return window" in result.steps[0].observation
    # Sources collected from observations.
    assert result.sources == ["returns.md"]
    # Terminated on Final Answer.
    assert result.stopped_reason == "final_answer"
    assert "30 days" in result.answer


def test_loop_caps_steps_when_model_never_finishes():
    """Model only ever emits actions => loop must stop at max_steps."""
    def never_finishes(_user_prompt: str) -> str:
        return "Thought: keep going\nAction: search\nAction Input: anything"

    result = agent.run_agent(
        "endless?",
        corpus=agent.Corpus(),
        llm_call=never_finishes,
        max_steps=3,
    )
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3  # exactly max_steps tool calls


def test_loop_dedupes_sources_across_steps():
    scripted = iter(
        [
            "Action: search\nAction Input: returns",
            "Action: search\nAction Input: returns again",
            "Final Answer: done (returns.md)",
        ]
    )

    class OneSourceCorpus(agent.Corpus):
        def search_text(self, query, top_k=3):
            return ("[returns.md] x", ["returns.md"])

    result = agent.run_agent(
        "q",
        corpus=OneSourceCorpus(),
        llm_call=lambda _p: next(scripted),
        max_steps=6,
    )
    assert result.sources == ["returns.md"]  # deduped


# --------------------------------------------------------------------------- #
# LLM helper
# --------------------------------------------------------------------------- #
def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("qwen-local-instruct", "x") == "x"
    assert llm.apply_no_think("claude-haiku-4-5", "x") == "x"


def test_supports_stop_is_false_only_for_claude_models():
    """The gateway's Anthropic path 500s on an OpenAI `stop` array."""
    assert llm.model_profile("qwen-local-instruct")["supports_stop"] is True
    assert llm.model_profile("qwen3:1.7b")["supports_stop"] is True
    assert llm.model_profile("claude-haiku")["supports_stop"] is False


def test_truncate_at_stop_cuts_hallucinated_observation():
    text = "Thought: look it up\nAction: search\nObservation: fabricated!"
    assert llm.truncate_at_stop(text) == "Thought: look it up\nAction: search"
    # Nothing to cut: returned unchanged (stripped).
    assert llm.truncate_at_stop("Thought: done  ") == "Thought: done"


def test_chat_omits_stop_for_claude_but_still_truncates():
    """Regression: sending `stop` to claude-* 500s, so the cut happens locally."""
    client = _fake_openai_client(["Action: search\nObservation: fake"])
    out = llm.chat(
        client,
        model="claude-haiku",
        system_prompt="s",
        user_prompt="u",
    )
    assert "stop" not in client.chat.completions.create.call_args.kwargs
    assert out == "Action: search"


def test_chat_sends_stop_for_models_that_support_it():
    client = _fake_openai_client(["Action: search"])
    llm.chat(client, model="qwen-local-instruct", system_prompt="s", user_prompt="u")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stop"] == list(llm.STOP_MARKERS)


# --------------------------------------------------------------------------- #
# HTTP level (mocked corpus + mocked openai client, no network)
# --------------------------------------------------------------------------- #
def _fake_openai_client(replies: list[str]) -> MagicMock:
    client = MagicMock()
    it = iter(replies)

    def _create(*_args, **_kwargs):
        msg = MagicMock()
        msg.content = next(it)
        choice = MagicMock()
        choice.message = msg
        return MagicMock(choices=[choice])

    client.chat.completions.create.side_effect = _create
    return client


def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(
        [
            "Thought: look up returns.\nAction: search\nAction Input: refund original packaging restocking",
            "Thought: now the warranty.\nAction: search\nAction Input: warranty defects coverage",
            "Thought: done.\nFinal Answer: 30-day returns; 1-year warranty "
            "(returns.md, warranty.md).",
        ]
    )
    app.state.corpus = agent.Corpus()

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "04-research-agent",
    }

    r = client.post("/run", json={"question": "returns and warranty?"})
    assert r.status_code == 200
    body = r.json()
    assert "warranty" in body["answer"].lower()
    assert "returns.md" in body["sources"]
    assert "warranty.md" in body["sources"]
    assert body["steps"], "expected at least one recorded step"
    assert body["steps"][0]["action"] == "search"
