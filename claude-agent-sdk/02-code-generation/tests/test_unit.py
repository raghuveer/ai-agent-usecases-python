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

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from app import codegen
from app.agent import AgentResult, ToolCall
from app.codegen import generate, sandbox_settings, sandbox_supported
from app.main import create_app
from app.settings import Settings

SOLUTION_SRC = "def add(a, b):\n    return a + b\n"
TEST_SRC = "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def make_runner(
    *,
    write_files: bool = True,
    used_bash: bool = True,
    pytest_ran: bool = True,
    is_error: bool = False,
    stop_reason: str | None = None,
):
    """Stub agent that simulates the built-in tools by writing into cwd.

    ``pytest_ran`` controls whether pytest's own ``.pytest_cache`` appears —
    i.e. whether the Bash calls actually *worked*. Defaulting it to True and
    ``used_bash`` to True keeps "the happy path" honest; setting it False
    reproduces a sandbox that refused every command.
    """

    async def runner(prompt, options) -> AgentResult:
        workdir = Path(options.cwd)
        if write_files:
            (workdir / "solution.py").write_text(SOLUTION_SRC, encoding="utf-8")
            (workdir / "test_solution.py").write_text(TEST_SRC, encoding="utf-8")
        if pytest_ran:
            (workdir / ".pytest_cache").mkdir(exist_ok=True)

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


def test_tests_passed_false_when_bash_ran_but_pytest_never_did():
    """The false positive this check exists for, reproduced.

    Observed in a container: the shell was sandboxed but bubblewrap could not
    start, so every Bash call died on `bwrap: pivot_root: Operation not
    permitted`. The agent retried, gave up, reasoned about its own source and
    declared the code correct — and `tests_passed` came back **true**, because
    the check only asked whether Bash had been *invoked*.

    Everything below is exactly what that run produced: both files written,
    Bash called, no error, a normal `end_turn`. The only thing missing is
    pytest's own footprint — which is the whole point.
    """
    body = make_client(pytest_ran=False).post("/run", json={"task": "x"}).json()
    assert body["files"] == ["solution.py", "test_solution.py"]
    assert "Bash" in body["tools_used"]
    assert body["stop_reason"] == "end_turn"
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
        # Write and Edit are available (`tools=`) but not auto-approved, so the
        # workdir gate sees them — see the F10 section below.
        assert set(options.tools) >= {"Write", "Bash"}
        assert "Bash" in options.allowed_tools
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


# --------------------------------------------------------------------------- #
# F9 -- the shell is sandboxed by default (docs/security-review.md)
# --------------------------------------------------------------------------- #
@pytest.fixture
def on_linux(monkeypatch):
    """Pretend the host supports sandboxing.

    These assertions are about the *policy*, which must hold everywhere — not
    about which OS the maintainer happens to use. Skipping them on Windows would
    leave a security control untested on the machine it is authored on.
    """
    monkeypatch.setattr(codegen, "sandbox_supported", lambda: True)


def test_sandbox_is_on_by_default(on_linux):
    """`Bash` here runs model output driven by untrusted request text. The
    sandbox must not be something a deployer has to remember to switch on."""
    assert sandbox_settings(Settings())["enabled"] is True


def test_sandbox_cannot_be_switched_off_by_the_agent(on_linux):
    """`allowUnsandboxedCommands` defaults to True in the SDK, which lets any
    command opt out via `dangerouslyDisableSandbox`. The thing choosing commands
    is model output -- a control the attacker can request be disabled is not a
    control."""
    assert sandbox_settings(Settings())["allowUnsandboxedCommands"] is False


def test_sandbox_denies_the_network_by_default(on_linux):
    """Writing and testing a function needs no network. Denying it removes the
    exfiltration half of a prompt-injection payload."""
    assert sandbox_settings(Settings())["network"] == {"allowedDomains": []}
    assert sandbox_settings(Settings(sandbox_allow_network=True))["network"] == {}


def test_sandbox_can_be_disabled_explicitly(on_linux):
    assert sandbox_settings(Settings(sandbox_bash=False)) is None


def test_sandbox_is_not_claimed_where_the_sdk_cannot_provide_it(monkeypatch):
    """On Windows the SDK accepts the setting and does nothing. Returning a
    config there would make `sandboxed` report a boundary that is not present —
    the one failure mode worse than having no sandbox."""
    monkeypatch.setattr(codegen, "sandbox_supported", lambda: False)
    assert sandbox_settings(Settings()) is None


# Verbatim from a live Windows run with the sandbox forced on — the case that
# proved config alone cannot answer "was this confined?".
CLI_DOWNGRADE_WARNING = (
    "⚠ Sandbox disabled: sandbox is enabled but the Windows sandbox is not "
    "active on this session (feature gate off)"
)


def test_monitor_reports_active_when_the_cli_says_nothing():
    monitor = codegen.SandboxMonitor(requested=True)
    monitor.observe("some unrelated cli chatter")
    assert monitor.active is True
    assert monitor.note is None


def test_monitor_believes_the_cli_over_the_config():
    """The bug found by running it: the sandbox was requested and accepted, the
    CLI downgraded it, and the response still claimed `sandboxed: true`."""
    monitor = codegen.SandboxMonitor(requested=True)
    monitor.observe(CLI_DOWNGRADE_WARNING)
    assert monitor.active is False
    assert monitor.note is not None and "feature gate off" in monitor.note


def test_monitor_is_never_active_when_no_sandbox_was_requested():
    assert codegen.SandboxMonitor(requested=False).active is False


@pytest.mark.anyio
async def test_run_wires_the_sandbox_and_the_stderr_observer_together():
    """A sandbox request without an observer would be a claim nobody checks."""
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["sandbox"] = options.sandbox
        seen["stderr"] = options.stderr
        # Speak as the CLI does when it cannot honour the request.
        if options.stderr is not None:
            options.stderr(CLI_DOWNGRADE_WARNING)
        return AgentResult(text="done")

    out = await generate("write add(a, b)", Settings(), spy)

    assert seen["stderr"] is not None, "no observer: `sandboxed` could only guess"
    assert (seen["sandbox"] is not None) is sandbox_supported()
    # The CLI said no, so the answer is no regardless of what was configured.
    assert out.sandboxed is False
    if sandbox_supported():
        assert "feature gate off" in out.sandbox_note


# --------------------------------------------------------------------------- #
# F10 — writes are confined to the workdir by a permission gate
# --------------------------------------------------------------------------- #
async def _decide(gate, tool: str, **inp):
    return await gate(tool, inp, ToolPermissionContext())


@pytest.fixture
def gate(tmp_path):
    return codegen.make_workdir_gate(tmp_path)


@pytest.mark.anyio
async def test_relative_writes_are_allowed(gate):
    for name in ("solution.py", "test_solution.py", "pkg/mod.py"):
        d = await _decide(gate, "Write", file_path=name, content="x")
        assert isinstance(d, PermissionResultAllow), name


@pytest.mark.anyio
async def test_absolute_write_outside_the_workdir_is_refused(gate):
    """The escape live runs actually attempted: `/tmp/solution.py`.

    `cwd` does not stop it — the Write tool takes absolute paths — and the
    prompt asking for relative names is guidance, not a boundary.
    """
    d = await _decide(gate, "Write", file_path="/tmp/solution.py", content="x")
    assert isinstance(d, PermissionResultDeny)
    assert "outside the working directory" in d.message
    # Not interrupted: the agent should fix the path and keep going.
    assert d.interrupt is False


@pytest.mark.anyio
async def test_traversal_out_of_the_workdir_is_refused(gate):
    d = await _decide(gate, "Write", file_path="../escaped.py", content="x")
    assert isinstance(d, PermissionResultDeny)


@pytest.mark.anyio
async def test_symlink_out_of_the_workdir_is_refused(tmp_path):
    """`resolve()` follows links, so a link planted inside cannot be a way out."""
    outside = tmp_path / "outside"
    outside.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    try:
        (workdir / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    d = await _decide(codegen.make_workdir_gate(workdir), "Write",
                      file_path="link/escaped.py", content="x")
    assert isinstance(d, PermissionResultDeny)


@pytest.mark.anyio
async def test_edit_is_gated_too_and_other_tools_pass_through(gate):
    assert isinstance(
        await _decide(gate, "Edit", file_path="/etc/passwd"), PermissionResultDeny
    )
    for tool in ("Read", "Glob", "Bash"):
        assert isinstance(await _decide(gate, tool, command="ls"), PermissionResultAllow)


@pytest.mark.anyio
async def test_write_without_a_path_is_refused(gate):
    assert isinstance(await _decide(gate, "Write", content="x"), PermissionResultDeny)


def test_write_tools_are_not_auto_approved():
    """The F11 lesson, asserted rather than remembered.

    Naming a tool in `allowed_tools` auto-approves it *before* `can_use_tool`
    runs. If Write or Edit ever reappears there, the gate above silently stops
    running and every test in this section still passes.
    """
    assert set(codegen.WRITE_TOOLS).isdisjoint(codegen.CODEGEN_AUTO_APPROVED)
    assert set(codegen.CODEGEN_AUTO_APPROVED) <= set(codegen.CODEGEN_TOOLS)


@pytest.mark.anyio
async def test_run_installs_the_gate_and_does_not_shadow_it():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["gate"] = options.can_use_tool
        seen["mode"] = options.permission_mode
        seen["allowed"] = options.allowed_tools
        return AgentResult(text="done")

    await generate("write add(a, b)", Settings(), spy)

    assert seen["gate"] is not None
    # "acceptEdits" would auto-accept writes before the gate is consulted.
    assert seen["mode"] == "default"
    assert "Write" not in seen["allowed"] and "Edit" not in seen["allowed"]
