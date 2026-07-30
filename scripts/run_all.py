#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
"""Run a command across every project (40 use cases + 4 templates).

The repo deliberately duplicates each project so it can be cloned in isolation.
The cost of that is fan-out: checking the repo means doing the same thing 44
times. These are the three that get done most.

    python scripts/run_all.py unit          # offline unit tests everywhere
    python scripts/run_all.py sync          # uv sync --extra dev --locked
    python scripts/run_all.py lock          # regenerate every uv.lock
    python scripts/run_all.py unit --only langgraph

Exits non-zero if any project fails, and always prints the roll-up — a partial
sweep that stops at the first failure hides how much else is broken.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APPROACHES = ("raw-api", "langchain", "langgraph", "claude-agent-sdk")
PASSED = re.compile(r"(\d+) passed")


def projects(only: str | None) -> list[Path]:
    found: list[Path] = []
    for approach in APPROACHES:
        if only and only != approach:
            continue
        for path in sorted((REPO / approach).iterdir()):
            if path.is_dir() and (path / "pyproject.toml").exists():
                found.append(path)
    return found


def venv_python(project: Path) -> Path | None:
    for candidate in (
        project / ".venv" / "Scripts" / "python.exe",
        project / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    return None


def command_for(action: str, project: Path) -> list[str] | None:
    if action in ("sync", "lock"):
        # `python -m uv` rather than a bare `uv`: uv is installed into the
        # environment here, not necessarily on PATH.
        args = ["sync", "--extra", "dev", "--locked"] if action == "sync" else ["lock"]
        return [sys.executable, "-m", "uv", *args]

    python = venv_python(project)
    if python is None:
        return None
    if not (project / "tests" / "test_unit.py").exists():
        return None
    return [
        str(python), "-m", "pytest", "tests/test_unit.py",
        "-q", "-p", "no:cacheprovider",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("unit", "sync", "lock"))
    parser.add_argument("--only", choices=APPROACHES, help="one approach only")
    args = parser.parse_args()

    started = time.perf_counter()
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []
    total_tests = 0

    for project in projects(args.only):
        name = project.relative_to(REPO).as_posix()
        command = command_for(args.action, project)
        if command is None:
            skipped.append(name)
            print(f"SKIP  {name} (no venv or no tests)")
            continue

        proc = subprocess.run(
            command, cwd=project, capture_output=True, text=True, timeout=1800
        )
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            tail = output.strip().splitlines()[-1:] or ["(no output)"]
            failures.append((name, tail[0]))
            print(f"FAIL  {name} :: {tail[0]}")
            continue

        match = PASSED.search(output)
        if match:
            total_tests += int(match.group(1))
            print(f"ok    {name} :: {match.group(0)}")
        else:
            print(f"ok    {name}")

    elapsed = int(time.perf_counter() - started)
    print(f"\n--- {args.action}: {elapsed}s ---")
    if total_tests:
        print(f"{total_tests} tests passed")
    if skipped:
        print(f"{len(skipped)} skipped")
    if failures:
        print(f"{len(failures)} FAILED:")
        for name, reason in failures:
            print(f"  {name} :: {reason}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
