# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC02 Code generation (claude-agent-sdk). See claude-agent-sdk/02-code-generation/README.md
"""Unit tests for UC02 code-generation (claude-agent-sdk). Stubbed agent, no network.

The stub runner writes real files into `options.cwd`, standing in for the built-in
Write/Bash tools. That exercises the parts this project actually owns: workdir
sandboxing, artefact collection, and the deliberately-conservative
`tests_passed` judgement.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentResult, ToolCall
from app.codegen import generate
from app.main import create_app
from app.settings import Settings

SOLUTION_SRC = "def add(a, b):\n    return a + b\n"
TEST_SRC = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def make_runner(
    *,
    write_files: bool = True,
    used_bash: bool = True,
    is_error: bool = False,
    stop_reason: str | None = None,
):
    """Stub agent that simulates the built-in tools by writing into cwd."""

    async def runner(prompt, options) -> AgentResult:
        if write_files:
            workdir = Path(options.cwd)
            (workdir / "solution.py").write_text(SOLUTION_SRC, encoding="utf-8")
            (workdir / "test_solution.py").write_text(TEST_SRC, encoding="utf-8")

        calls = [ToolCall(name="Write"), ToolCall(name="Write")]
        if used_bash:
            calls.append(ToolCall(name="Bash", input={"command": "python -m pytest -q"}))

        return AgentResult(
            text="Tests pass.",
            tool_calls=calls,
            num_turns=4,
            cost_usd=0.01,
            is_error=is_error,
            stop_reason=stop_reason,
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
        "usecase": "02-code-generation",
    }


def test_run_returns_generated_artifacts():
    resp = make_client().post("/run", json={"task": "Write an add(a, b) function."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["solution"] == SOLUTION_SRC
    assert body["tests"] == TEST_SRC
    assert body["tests_passed"] is True
    assert sorted(body["files"]) == ["solution.py", "test_solution.py"]
    assert "Bash" in body["tools_used"]
    assert body["num_turns"] == 4


def test_tests_passed_false_when_agent_never_ran_bash():
    """Files alone are not evidence — the agent must have executed the tests."""
    body = make_client(used_bash=False).post("/run", json={"task": "x"}).json()
    assert body["tests_passed"] is False


def test_tests_passed_false_when_files_missing():
    body = make_client(write_files=False).post("/run", json={"task": "x"}).json()
    assert body["tests_passed"] is False
    assert body["solution"] == ""
    assert body["files"] == []


def test_tests_passed_false_on_turn_limit():
    body = make_client(stop_reason="max_turns").post("/run", json={"task": "x"}).json()
    assert body["tests_passed"] is False


def test_tests_passed_false_on_error_result():
    body = make_client(is_error=True).post("/run", json={"task": "x"}).json()
    assert body["tests_passed"] is False


@pytest.mark.anyio
async def test_workdir_is_isolated_and_cleaned_up():
    """Each run gets a fresh temp dir, and it does not survive the call."""
    seen: list[Path] = []

    async def spy(prompt, options) -> AgentResult:
        seen.append(Path(options.cwd))
        return AgentResult(text="ok")

    settings = Settings()
    await generate("task one", settings, spy)
    await generate("task two", settings, spy)

    assert len(seen) == 2
    assert seen[0] != seen[1], "runs must not share a workdir"
    assert not seen[0].exists() and not seen[1].exists(), "workdirs must be removed"


@pytest.mark.anyio
async def test_agent_is_confined_to_the_workdir_and_offline_tools():
    """cwd scopes the agent; no web tools are on the allow-list."""

    async def spy(prompt, options) -> AgentResult:
        assert options.cwd is not None
        assert "WebFetch" not in options.allowed_tools
        assert "WebSearch" not in options.allowed_tools
        assert set(options.allowed_tools) >= {"Write", "Bash"}
        assert options.permission_mode == "acceptEdits"
        return AgentResult(text="ok")

    await generate("task", Settings(), spy)


@pytest.mark.anyio
async def test_cap_exhaustion_is_reported_not_raised(monkeypatch):
    """Turn/budget caps are conditions we configured, so they must not 500.

    The SDK raises on terminal conditions instead of yielding a ResultMessage.
    Regression for a bug the live integration test caught.
    """
    import app.agent as agent_mod
    from claude_agent_sdk import ClaudeAgentOptions

    for msg, expected in (
        ("Claude Code returned an error result: Reached maximum number of turns (12)", "max_turns"),
        ("Claude Code returned an error result: Reached maximum budget ($0.25)", "max_budget"),
    ):
        def fake_query(*, prompt, options, _m=msg):
            async def _boom():
                raise Exception(_m)
                yield  # pragma: no cover
            return _boom()

        monkeypatch.setattr(agent_mod, "query", fake_query)
        out = await agent_mod.default_runner("task", ClaudeAgentOptions())
        assert out.is_error is True
        assert out.stop_reason == expected


@pytest.mark.anyio
async def test_genuine_errors_still_propagate(monkeypatch):
    import app.agent as agent_mod
    from claude_agent_sdk import ClaudeAgentOptions

    def fake_query(*, prompt, options):
        async def _boom():
            raise Exception("connection refused")
            yield  # pragma: no cover
        return _boom()

    monkeypatch.setattr(agent_mod, "query", fake_query)
    with pytest.raises(Exception, match="connection refused"):
        await agent_mod.default_runner("task", ClaudeAgentOptions())


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"task": "write add(a, b)"}


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
