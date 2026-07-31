# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC04 Research agent (claude-agent-sdk). See claude-agent-sdk/04-research-agent/README.md
"""Unit tests for UC04 research-agent (claude-agent-sdk). Stubbed agent, no network.

The important assertion is the air-gap guarantee: in the default offline mode the
web tools must not be on the allow-list at all, so an agent physically cannot
reach the network.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from app import research as MODULE
from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.research import CORPUS_DIR, research
from app.settings import Settings

ANSWER = "Local Qwen models could not sustain a text ReAct loop."


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=ANSWER,
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name="Grep", input={"pattern": "ReAct"}),
                ToolCall(name="Read", input={"file_path": "/c/model-findings.md"}),
            ],
            num_turns=4,
            cost_usd=0.003,
        )

    return runner


def make_client(**kw) -> TestClient:
    return TestClient(create_app(runner=make_runner(**kw)))


def test_health_reports_mode():
    body = make_client().get("/health").json()
    assert body["status"] == "ok"
    assert body["approach"] == "claude-agent-sdk"
    assert body["usecase"] == "04-research-agent"
    assert body["mode"] == "offline"


def test_run_returns_answer_with_citations_and_searches():
    body = make_client().post("/run", json={"question": "Can Qwen do ReAct?"}).json()
    assert body["answer"] == ANSWER
    assert body["mode"] == "offline"
    assert body["citations"] == ["model-findings.md"]
    assert body["searches"] == ["ReAct"]


def test_citations_include_urls_when_web_tools_were_used():
    calls = [
        ToolCall(name="WebSearch", input={"query": "react agents"}),
        ToolCall(name="WebFetch", input={"url": "https://example.com/a"}),
        ToolCall(name="Read", input={"file_path": "/c/local.md"}),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == ["https://example.com/a", "local.md"]
    assert body["searches"] == ["react agents"]


def test_citations_are_deduped_in_first_use_order():
    calls = [
        ToolCall(name="Read", input={"file_path": "/c/b.md"}),
        ToolCall(name="Read", input={"file_path": "/c/a.md"}),
        ToolCall(name="Read", input={"file_path": "/c/b.md"}),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == ["b.md", "a.md"]


def test_citations_come_from_tool_calls_not_prose():
    """A source the agent never opened must not be reported as a citation."""
    calls = [ToolCall(name="Grep", input={"pattern": "x"})]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["citations"] == []


@pytest.mark.anyio
async def test_offline_mode_cannot_reach_the_network():
    """The air-gap guarantee: web tools are absent from the allow-list."""
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        # `tools=` is what the agent HAS; `allowed_tools` is only what is
        # auto-approved, and is deliberately empty so the corpus gate runs.
        seen["tools"] = options.tools
        seen["prompt"] = options.system_prompt
        seen["cwd"] = options.cwd
        return AgentResult(text="ok")

    await research("q", Settings(research_allow_web=False), spy)

    assert set(seen["tools"]) == {"Grep", "Glob", "Read"}
    assert "WebSearch" not in seen["tools"] and "WebFetch" not in seen["tools"]
    assert "NO internet access" in seen["prompt"]
    assert Path(seen["cwd"]) == CORPUS_DIR


@pytest.mark.anyio
async def test_web_mode_adds_web_tools_and_keeps_local_ones():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        # `tools=` is what the agent HAS; `allowed_tools` is only what is
        # auto-approved, and is deliberately empty so the corpus gate runs.
        seen["tools"] = options.tools
        seen["prompt"] = options.system_prompt
        return AgentResult(text="ok")

    await research("q", Settings(research_allow_web=True), spy)

    assert set(seen["tools"]) == {"Grep", "Glob", "Read", "WebSearch", "WebFetch"}
    assert "may search the web" in seen["prompt"]


@pytest.mark.anyio
async def test_research_raises_turn_ceiling_for_iteration():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["max_turns"] = options.max_turns
        return AgentResult(text="ok")

    await research("q", Settings(), spy)
    assert seen["max_turns"] >= 8


def test_bundled_corpus_exists():
    assert list(CORPUS_DIR.glob("*.md")), f"no corpus documents in {CORPUS_DIR}"


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"question": "q"}


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


# --------------------------------------------------------------------------- #
# F15 — file access is confined to the corpus
# --------------------------------------------------------------------------- #
async def _decide(gate, tool="Read", **inp):
    return await gate(tool, inp, ToolPermissionContext())


@pytest.fixture
def corpus_gate(tmp_path):
    return MODULE.make_corpus_gate(tmp_path)


@pytest.mark.anyio
async def test_reads_inside_the_corpus_are_allowed(corpus_gate, tmp_path):
    """The shapes the CLI actually sends: absolute, relative, and no path."""
    for inp in (
        {"file_path": str(tmp_path / "notes.md")},
        {"file_path": "notes.md"},
        {"path": str(tmp_path)},
        {"path": "."},
        {"pattern": "ReAct"},
    ):
        assert isinstance(await _decide(corpus_gate, **inp), PermissionResultAllow), inp


@pytest.mark.anyio
async def test_paths_outside_the_corpus_are_refused(corpus_gate, tmp_path):
    """The attack that worked before this gate existed.

    A live agent asked to read an absolute path outside its corpus did so and
    reported success — including a project's own `.env`, which holds the gateway
    key. "Read /path/to/.env" is the whole technique.
    """
    for raw in ("/etc/passwd", "../secrets.env", str(tmp_path.parent / "x.md")):
        d = await _decide(corpus_gate, file_path=raw)
        assert isinstance(d, PermissionResultDeny), raw
        assert d.interrupt is False


@pytest.mark.anyio
async def test_symlink_out_of_the_corpus_is_refused(tmp_path):
    """`resolve()` follows links, so one planted inside is not a way out."""
    outside, corpus = tmp_path / "outside", tmp_path / "corpus"
    outside.mkdir(); corpus.mkdir()
    try:
        (corpus / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    d = await _decide(MODULE.make_corpus_gate(corpus), file_path="link/secrets.env")
    assert isinstance(d, PermissionResultDeny)


@pytest.mark.anyio
async def test_every_path_argument_spelling_is_checked(corpus_gate):
    """Read says file_path, Grep/Glob say path. Missing one leaves a hole."""
    for key in MODULE.PATH_ARGS:
        assert isinstance(
            await _decide(corpus_gate, **{key: "/etc/passwd"}), PermissionResultDeny
        ), key


@pytest.mark.anyio
async def test_tools_without_a_path_are_not_blocked(corpus_gate):
    """Delegation and web tools carry no path; gating them would break the run."""
    for tool, inp in (("Agent", {"subagent_type": "researcher"}),
                      ("WebFetch", {"url": "https://example.com"})):
        assert isinstance(await _decide(corpus_gate, tool, **inp), PermissionResultAllow)
