# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (raw-api). See raw-api/02-code-generation/README.md
"""Hand-written code generation: prompt -> generate -> extract -> (optional) check.

This is the raw-api point: nothing is hidden behind a framework. We build the
prompt by hand, call the coder model, pull the code out of the first fenced
block, and (only when explicitly enabled) run a single safe smoke check in a
subprocess. The LLM call is injectable so unit tests never touch the network and
never execute code.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

# An LLM call is `(system_prompt, user_prompt) -> text`. Injectable for tests.
LLMCall = Callable[[str, str], str]

SYSTEM_PROMPT = (
    "You are an expert programmer. Generate correct, self-contained code for the "
    "user's task. Respond with exactly ONE fenced code block in the requested "
    "language, followed by a short plain-text explanation. Do not include any "
    "prose before the code block."
)

# Matches the first ```lang ... ``` fenced block; the language tag is optional.
_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def build_prompt(task: str, language: str) -> str:
    """Build the grounded user prompt for the coder model."""
    return (
        f"Task: {task}\n\n"
        f"Language: {language}\n\n"
        f"Write the {language} code that accomplishes the task. Return one "
        f"```{language} code block, then one or two sentences explaining it."
    )


def extract_code(text: str) -> str:
    """Pull the code out of the first fenced block.

    Falls back to the whole stripped text when no fence is present, so a model
    that forgets the fences still yields usable code.
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(2).strip()
    return text.strip()


def extract_explanation(text: str) -> str:
    """Return any prose that follows the first fenced code block."""
    match = _FENCE_RE.search(text)
    if not match:
        return ""
    return text[match.end():].strip()


def smoke_check_python(code: str, timeout: int) -> bool:
    """Run ``code`` once in a subprocess; True if it executes without error.

    This is a trivial smoke: it confirms the code at least imports/executes. It
    is OFF by default and only ever called when the caller opts in via the
    ``run_code_check`` setting. We run the current interpreter on a temp file
    and treat a zero exit (within the timeout) as a pass.

    ⚠️ SECURITY — ARBITRARY CODE EXECUTION. ``code`` is produced by the LLM and
    is therefore UNTRUSTED. Running it executes whatever the model emitted with
    the full privileges of this process: it can read/write/delete files, open
    network connections, exfiltrate environment variables and secrets, spawn
    further processes, or exhaust resources. The ``timeout`` here is the ONLY
    guardrail and bounds wall-clock time alone — it does not sandbox the code,
    restrict filesystem/network access, or cap memory. For this reason the
    feature is gated behind ``RUN_CODE_CHECK`` (OFF by default).

    DO NOT enable this directly on a host you care about. If you turn it on, run
    it inside a disposable, isolated sandbox: a throwaway container or VM with no
    network, strict CPU/memory/time and filesystem limits, an unprivileged user,
    and NO secrets or credentials mounted on the host. Treat any output as
    untrusted.
    """
    tmp = Path(tempfile.mkdtemp()) / "candidate.py"
    tmp.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def generate(
    task: str,
    *,
    language: str,
    llm_call: LLMCall,
    run_code_check: bool = False,
    code_check_timeout: int = 10,
    max_iters: int = 2,
) -> dict:
    """Generate code for ``task``.

    When ``run_code_check`` is True and the language is python, the generated
    code is smoke-run in a subprocess; on failure we iterate (asking the model to
    fix it), capped at ``max_iters`` attempts. When the flag is off no code is
    ever executed and ``tests_passed`` is ``None``.
    """
    user_prompt = build_prompt(task, language)
    raw = llm_call(SYSTEM_PROMPT, user_prompt)
    code = extract_code(raw)
    explanation = extract_explanation(raw)

    tests_passed: bool | None = None
    if run_code_check and language == "python":
        for attempt in range(max_iters):
            tests_passed = smoke_check_python(code, code_check_timeout)
            if tests_passed or attempt == max_iters - 1:
                break
            # Iterate-on-failure: ask the model to fix the broken code.
            fix_prompt = (
                f"{user_prompt}\n\nThe previous attempt failed to run:\n"
                f"```{language}\n{code}\n```\n"
                "Return a corrected single fenced code block."
            )
            raw = llm_call(SYSTEM_PROMPT, fix_prompt)
            code = extract_code(raw)
            explanation = extract_explanation(raw)

    return {
        "code": code,
        "language": language,
        "explanation": explanation,
        "tests_passed": tests_passed,
    }
