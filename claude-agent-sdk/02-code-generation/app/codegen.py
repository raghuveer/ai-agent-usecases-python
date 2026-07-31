# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC02 Code generation (claude-agent-sdk). See claude-agent-sdk/02-code-generation/README.md
"""Code generation as a real write-run-fix loop, using the SDK's built-in tools.

This is the Agent SDK's home turf, and the contrast with the other three
approaches is stark. There, "generate code and check it" means *you* write the
subprocess sandbox, the timeout, the test-output parser, and the retry loop. Here
the agent already has `Write`, `Read`, `Edit`, and `Bash`; you hand it a working
directory and a definition of done, and it iterates on its own:

    write solution.py ──► write test_solution.py ──► Bash: pytest
             ▲                                          │
             └───────────── fix on failure ◄────────────┘

The only code below is sandboxing (a per-run temp dir), the prompt, and reading
the artefacts back out. There is no loop — `max_turns` bounds it.

**Sandboxing — `cwd` is NOT a sandbox.** This was verified the hard way against a
live agent: `cwd` sets the working directory, but `Write` happily accepts an
*absolute* path, and the model repeatedly chose `/tmp/solution.py` instead of the
workdir. It is not even reliable containment, let alone security — and `Bash` can
run anything the server user can. Two mitigations are applied here:

1. The system prompt explicitly demands bare relative filenames and forbids
   absolute paths and `/tmp`.
2. Artefacts are only ever read back from the workdir, so anything written
   outside it simply does not count as output (`tests_passed` stays false).

Neither is a security boundary. So a third mitigation now applies by default:

3. The SDK's `sandbox` setting confines `Bash` — no network, and no way for a
   command to opt itself out (`allowUnsandboxedCommands: False`). See
   :func:`sandbox_settings`.

That one *is* a boundary, with a caveat worth stating plainly: the SDK sandboxes
bash on **macOS/Linux only**. On Windows the setting is accepted and silently
does nothing, so every response reports `sandboxed` — what actually happened —
rather than what was asked for. When it is false, running untrusted task text
still needs a disposable, network-isolated container or VM.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentResult, Runner, build_options, default_runner, outcome_of
from .settings import Settings

SYSTEM_PROMPT = """You are a Python engineer working in the current directory.

Given a task:
1. Write the implementation to `solution.py`.
2. Write pytest tests to `test_solution.py` that genuinely exercise the task,
   including edge cases.
3. Run `python -m pytest test_solution.py -q` with the Bash tool.
4. If tests fail, fix the code and re-run until they pass.

PATHS: use the bare relative filenames `solution.py` and `test_solution.py`,
exactly as written. Never use an absolute path and never write to /tmp — the
current directory is already a private scratch directory created for this task.

Only these two files. No packages beyond the standard library and pytest.
Stop as soon as the tests pass, and say so."""

# Built-in tools only — no custom tools needed, which is itself the point.
CODEGEN_TOOLS = ["Write", "Read", "Edit", "Bash", "Glob"]

SOLUTION = "solution.py"
TESTS = "test_solution.py"


@dataclass
class CodegenResult:
    """Artefacts recovered from the workdir plus the agent's own summary."""

    solution: str
    tests: str
    summary: str
    tests_passed: bool
    files: list[str]
    tool_calls: list[str]
    num_turns: int
    cost_usd: float
    stop_reason: str = "end_turn"
    # Whether the shell was actually confined for this run — the *observed*
    # state, not the requested one. See :class:`SandboxMonitor`.
    sandboxed: bool = False
    # Why not, when the CLI downgraded the request. Null when there was nothing
    # to report (sandbox active, or never asked for).
    sandbox_note: str | None = None


PYTEST_CACHE = ".pytest_cache"


def _ran_tests_successfully(result: AgentResult, workdir: Path) -> bool:
    """Did the agent actually get a green run?

    Both artefacts must exist, the agent must have invoked Bash, **pytest must
    have left its own cache directory in the workdir**, and the run must not
    have ended in an error or a cap.

    The ``.pytest_cache`` check exists because the rest was not enough. Running
    this in a container with the shell sandboxed but bubblewrap unable to start,
    every Bash call failed with ``bwrap: pivot_root: Operation not permitted``;
    the agent retried eight times, gave up, reasoned about its own source, and
    declared the implementation correct. This function returned **True** — it
    had asked whether Bash was *invoked*, never whether it *worked*, and the
    docstring's promise that the agent "really executed the tests rather than
    just asserting they would pass" was exactly what failed to hold.

    ``.pytest_cache`` is written by pytest itself, in the directory it ran in,
    so it is evidence from the tool rather than a claim from the agent.

    **What it still cannot prove is that the tests passed.** ``lastfailed``
    survives a subsequent green run when test ids change, so it is not a usable
    signal, and the only authoritative check — running the tests here — would
    execute model-written code *outside* the sandbox the agent's own shell runs
    in, trading a real boundary for a better status field. So this stays
    evidence-based: it now proves the tests ran, and it no longer mistakes an
    agent that never ran them for one that did.
    """
    if not (workdir / SOLUTION).exists() or not (workdir / TESTS).exists():
        return False
    if "Bash" not in result.tool_names:
        return False
    if not (workdir / PYTEST_CACHE).is_dir():
        return False
    return not result.is_error and result.stop_reason != "max_turns"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --------------------------------------------------------------------------- #
# F9 mitigation — sandbox the shell (docs/security-review.md §11.2)
# --------------------------------------------------------------------------- #
def sandbox_supported() -> bool:
    """Whether the SDK can actually sandbox bash on this host.

    The SDK documents bash sandboxing as **macOS/Linux only**. On Windows the
    setting is accepted and does nothing, which is the dangerous kind of
    failure: you would believe the shell was confined. Callers get the real
    answer in ``CodegenResult.sandboxed`` rather than the requested one.
    """
    return sys.platform != "win32"


def sandbox_settings(settings: Settings) -> dict[str, Any] | None:
    """Sandbox config for the run, or ``None`` when sandboxing is off/unavailable."""
    if not settings.sandbox_bash or not sandbox_supported():
        return None
    return {
        "enabled": True,
        # The agent is *supposed* to run pytest; prompting per command would
        # break the write-run-fix loop. This is only acceptable because what is
        # being auto-approved is a *sandboxed* command.
        "autoAllowBashIfSandboxed": True,
        # Load-bearing. Left at its default (True), any command can opt itself
        # out via `dangerouslyDisableSandbox` — and the thing choosing commands
        # here is model output driven by untrusted request text. A control the
        # attacker can ask to have switched off is not a control.
        "allowUnsandboxedCommands": False,
        "network": (
            {} if settings.sandbox_allow_network else {"allowedDomains": []}
        ),
    }


class SandboxMonitor:
    """Believes the CLI, not the config.

    Requesting a sandbox and getting one are different things, and the gap was
    found by running it: with the sandbox forced on under Windows the CLI
    printed

        ``⚠ Sandbox disabled: ... the Windows sandbox is not active on this
        session (feature gate off). Commands will run WITHOUT sandboxing.``

    — while the response still said ``sandboxed: true``, because the code was
    reporting what it had *asked for*. Claiming a boundary that is not there is
    worse than having none: it is the claim a deployer would act on.

    The CLI announces the downgrade on stderr, and the SDK forwards stderr to a
    callback, so the honest answer is available for the asking. This is the same
    lesson as ``setting_sources=[]`` (F14) — a security setting is only worth
    what its *observed* effect is.
    """

    _DISABLED_MARKERS = ("Sandbox disabled", "WITHOUT sandboxing")

    def __init__(self, requested: bool) -> None:
        self.requested = requested
        self.note: str | None = None

    def observe(self, line: str) -> None:
        if self.note is None and any(m in line for m in self._DISABLED_MARKERS):
            self.note = line.strip()

    @property
    def active(self) -> bool:
        """True only when a sandbox was requested *and* not downgraded."""
        return self.requested and self.note is None


async def generate(
    task: str,
    settings: Settings,
    runner: Runner | None = None,
    keep_workdir: bool = False,
) -> CodegenResult:
    """Run the agent in a fresh temp workdir and collect what it produced."""
    runner = runner or default_runner
    workdir = Path(tempfile.mkdtemp(prefix="agentsdk-codegen-"))
    sandbox = sandbox_settings(settings)
    monitor = SandboxMonitor(requested=sandbox is not None)
    try:
        options = build_options(
            settings,
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=CODEGEN_TOOLS,
            tools=CODEGEN_TOOLS,
            cwd=str(workdir),
            # Edits are auto-accepted: the agent owns this throwaway directory.
            permission_mode="acceptEdits",
            sandbox=sandbox,
            # Not logging — this is how the CLI reports that it could not honour
            # the sandbox. See SandboxMonitor.
            stderr=monitor.observe,
        )
        result = await runner(task, options)

        return CodegenResult(
            sandboxed=monitor.active,
            sandbox_note=monitor.note,
            solution=_read(workdir / SOLUTION),
            tests=_read(workdir / TESTS),
            summary=result.text,
            tests_passed=_ran_tests_successfully(result, workdir),
            files=sorted(p.name for p in workdir.iterdir() if p.is_file()),
            tool_calls=result.tool_names,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            stop_reason=outcome_of(result),
        )
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
