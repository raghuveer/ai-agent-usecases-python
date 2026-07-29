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

Neither is a security boundary. Running untrusted task text against this in
production needs a real sandbox (container/VM), or the SDK's `sandbox` setting.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentResult, Runner, build_options, default_runner
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


def _ran_tests_successfully(result: AgentResult, workdir: Path) -> bool:
    """Did the agent actually get a green run?

    Deliberately conservative: both artefacts must exist *and* the agent must
    have invoked Bash at least once (i.e. it really executed the tests rather
    than just asserting they would pass), and the run must not have ended in an
    error or a turn-limit stop.
    """
    if not (workdir / SOLUTION).exists() or not (workdir / TESTS).exists():
        return False
    if "Bash" not in result.tool_names:
        return False
    return not result.is_error and result.stop_reason != "max_turns"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def generate(
    task: str,
    settings: Settings,
    runner: Runner | None = None,
    keep_workdir: bool = False,
) -> CodegenResult:
    """Run the agent in a fresh temp workdir and collect what it produced."""
    runner = runner or default_runner
    workdir = Path(tempfile.mkdtemp(prefix="agentsdk-codegen-"))
    try:
        options = build_options(
            settings,
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=CODEGEN_TOOLS,
            tools=CODEGEN_TOOLS,
            cwd=str(workdir),
            # Edits are auto-accepted: the agent owns this throwaway directory.
            permission_mode="acceptEdits",
        )
        result = await runner(task, options)

        return CodegenResult(
            solution=_read(workdir / SOLUTION),
            tests=_read(workdir / TESTS),
            summary=result.text,
            tests_passed=_ran_tests_successfully(result, workdir),
            files=sorted(p.name for p in workdir.iterdir() if p.is_file()),
            tool_calls=result.tool_names,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
        )
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
