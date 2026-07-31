# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""Unit tests for UC07 multi-agent (claude-agent-sdk). Stubbed agent, no network.

Covers what this project owns: the roster definition (including the
least-privilege tool split, which is the interesting claim), and reading the
delegation trace back out of `Agent` tool calls.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from app import team as MODULE
from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.settings import Settings
from app.team import TEAM, run_team

REPORT = "Summary: local models cannot drive text ReAct loops reliably."


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=REPORT,
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name="Agent", input={"subagent_type": "researcher"}),
                ToolCall(name="Agent", input={"subagent_type": "analyst"}),
                ToolCall(name="Agent", input={"subagent_type": "writer"}),
            ],
            num_turns=7,
            cost_usd=0.03,
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
        "usecase": "07-multi-agent",
    }


def test_run_returns_report_and_delegation_trace():
    body = make_client().post("/run", json={"question": "What did we learn?"}).json()
    assert body["report"] == REPORT
    assert body["subagents_used"] == ["researcher", "analyst", "writer"]
    assert body["num_turns"] == 7


def test_delegation_trace_ignores_non_delegation_tools():
    calls = [
        ToolCall(name="Grep", input={"pattern": "x"}),
        ToolCall(name="Agent", input={"subagent_type": "researcher"}),
        ToolCall(name="Read", input={"file_path": "a.md"}),
    ]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["subagents_used"] == ["researcher"]
    assert body["tools_used"] == ["Grep", "Agent", "Read"]


def test_delegation_trace_tolerates_delegation_without_subagent_type():
    calls = [ToolCall(name="Agent", input={})]
    body = make_client(tool_calls=calls).post("/run", json={"question": "q"}).json()
    assert body["subagents_used"] == []


def test_team_endpoint_exposes_roster():
    body = make_client().get("/team").json()
    assert set(body) == {"researcher", "analyst", "writer"}


def test_roster_enforces_least_privilege():
    """The point of per-subagent tool lists: only the researcher touches files."""
    assert TEAM["researcher"].tools == ["Grep", "Glob", "Read"]
    assert TEAM["analyst"].tools == []
    assert TEAM["writer"].tools == []
    for name, definition in TEAM.items():
        assert "Write" not in (definition.tools or []), f"{name} must not write"
        assert "Bash" not in (definition.tools or []), f"{name} must not run shell"


@pytest.mark.anyio
async def test_options_register_subagents_and_raise_turn_ceiling():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["agents"] = options.agents
        # `tools=` is what the agent HAS; `allowed_tools` is only what is
        # auto-approved, and is deliberately empty so the corpus gate runs.
        seen["tools"] = options.tools
        seen["max_turns"] = options.max_turns
        return AgentResult(text="ok")

    await run_team("q", Settings(), spy)

    assert set(seen["agents"]) == {"researcher", "analyst", "writer"}
    assert "Agent" in seen["tools"], "delegation needs the Agent tool"
    # Fan-out needs more headroom than the shared default. 12 proved too tight
    # in live runs — three delegations plus the lead's own turns hit the cap and
    # the report came back empty — so the floor is 20.
    assert seen["max_turns"] >= 20


@pytest.mark.anyio
async def test_capped_run_still_surfaces_partial_report():
    """A capped run must report what the team produced, not an empty string.

    Regression for the live flake: the runner discarded partial work on a cap,
    so `/run` answered 200 with an empty report and no delegations even though
    the subagents had done the work.
    """

    async def capped(prompt, options) -> AgentResult:
        return AgentResult(
            text="Partial findings so far.",
            tool_calls=[ToolCall(name="Agent", input={"subagent_type": "researcher"})],
            num_turns=20,
            is_error=True,
            stop_reason="max_turns",
        )

    out = await run_team("q", Settings(), capped)

    assert out.report == "Partial findings so far."
    assert out.subagents_used == ["researcher"]


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
