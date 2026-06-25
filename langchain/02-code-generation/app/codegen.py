# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langchain). See langchain/02-code-generation/README.md
"""Code generation as an LCEL chain: prompt | llm | parse -> (optional) check.

The generation step is a plain LCEL pipeline so the pieces stay visible:

    ChatPromptTemplate | llm | StrOutputParser

``generate`` runs that chain, extracts the first fenced code block, and (only
when explicitly enabled) runs a single safe smoke check in a subprocess. The LLM
is injected so unit tests swap in ``FakeListChatModel`` and never touch the
network or execute code.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm import model_profile

SYSTEM_PROMPT = (
    "You are an expert programmer. Generate correct, self-contained code for the "
    "user's task. Respond with exactly ONE fenced code block in the requested "
    "language, followed by a short plain-text explanation. Do not include any "
    "prose before the code block."
)

USER_PROMPT = (
    "Task: {task}\n\n"
    "Language: {language}\n\n"
    "Write the {language} code that accomplishes the task. Return one "
    "```{language} code block, then one or two sentences explaining it."
)

# Matches the first ```lang ... ``` fenced block; the language tag is optional.
_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def _system_prompt(model: str) -> str:
    """Disable qwen3 thinking mode by prefixing /no_think."""
    return model_profile(model)["thinking_prefix"] + SYSTEM_PROMPT


def build_chain(llm: BaseChatModel, model_name: str):
    """Assemble the LCEL generation chain ({task, language} -> raw text)."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_prompt(model_name)), ("human", USER_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


def extract_code(text: str) -> str:
    """Pull the code out of the first fenced block; fall back to whole text."""
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

    Trivial smoke check (imports/executes). OFF by default; only ever called
    when the caller opts in. Zero exit within the timeout is a pass.

    ⚠️ SECURITY: ``code`` here is model-generated. Executing it is *arbitrary
    code execution* — there is no sandboxing, allow-listing, or syscall
    filtering in this function. The subprocess inherits this process's user,
    filesystem access, environment variables (including any secrets/API keys),
    and network access; the only guard is a wall-clock ``timeout``. This is why
    the check is OFF by default (``run_code_check=False``).

    Do NOT enable this against untrusted or production input as-is. If you opt
    in, run it inside a disposable, isolated sandbox: a throwaway
    container/VM/microVM (e.g. gVisor, Firecracker), with no network, no host
    secrets in the environment, a non-privileged user, a read-only/ephemeral
    filesystem, and CPU/memory/process/file-descriptor limits in addition to the
    time limit.
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
    llm: BaseChatModel,
    model_name: str,
    run_code_check: bool = False,
    code_check_timeout: int = 10,
    max_iters: int = 2,
) -> dict:
    """Generate code for ``task`` via the LCEL chain.

    When ``run_code_check`` is True and language==python, smoke-run the code and
    iterate on failure (capped at ``max_iters``). When off, nothing is executed
    and ``tests_passed`` is ``None``.
    """
    chain = build_chain(llm, model_name)
    raw = chain.invoke({"task": task, "language": language})
    code = extract_code(raw)
    explanation = extract_explanation(raw)

    tests_passed: bool | None = None
    if run_code_check and language == "python":
        for attempt in range(max_iters):
            tests_passed = smoke_check_python(code, code_check_timeout)
            if tests_passed or attempt == max_iters - 1:
                break
            # Iterate-on-failure: re-run the chain with the failing code appended.
            raw = chain.invoke(
                {
                    "task": (
                        f"{task}\n\nThe previous attempt failed to run:\n"
                        f"```{language}\n{code}\n```\nReturn a corrected version."
                    ),
                    "language": language,
                }
            )
            code = extract_code(raw)
            explanation = extract_explanation(raw)

    return {
        "code": code,
        "language": language,
        "explanation": explanation,
        "tests_passed": tests_passed,
    }
