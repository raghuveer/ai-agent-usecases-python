r"""Code generation as a langgraph StateGraph: generate -> check -> (loop|END).

This is where langgraph earns its keep over a flat chain: the OPTIONAL
self-check / fix loop is a real cyclic graph. The graph is:

    generate --> check --(fail & under cap)--> generate
                       \--(pass | flag off | cap)--> END

Both the LLM and the code executor are injected, so unit tests swap in fakes and
run fully offline and never execute generated code. When the self-check flag is
off, the ``check`` node is a no-op pass-through (``tests_passed`` stays None).
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import system_prompt
from .settings import Settings, get_settings

SYSTEM_INSTRUCTION = (
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


def build_user_prompt(task: str, language: str) -> str:
    return (
        f"Task: {task}\n\n"
        f"Language: {language}\n\n"
        f"Write the {language} code that accomplishes the task. Return one "
        f"```{language} code block, then one or two sentences explaining it."
    )


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


# --- Graph -----------------------------------------------------------------

class CodeGenState(TypedDict):
    task: str
    language: str
    raw: str
    code: str
    explanation: str
    tests_passed: bool | None
    run_code_check: bool
    iterations: int
    max_iters: int


def build_codegen_graph(
    llm: BaseChatModel,
    settings: Settings | None = None,
    *,
    executor=smoke_check_python,
):
    """Compile a generate -> check -> (loop|END) graph. Deps are injected.

    ``executor`` is the code smoke-runner; tests pass a fake so nothing runs.
    """
    settings = settings or get_settings()
    timeout = settings.code_check_timeout

    def generate(state: CodeGenState) -> dict:
        prompt = build_user_prompt(state["task"], state["language"])
        if state["iterations"] > 0 and state.get("code"):
            # Iterate-on-failure: include the failing code so the model fixes it.
            prompt += (
                f"\n\nThe previous attempt failed to run:\n"
                f"```{state['language']}\n{state['code']}\n```\n"
                "Return a corrected single fenced code block."
            )
        messages = [
            SystemMessage(content=system_prompt(SYSTEM_INSTRUCTION, settings)),
            HumanMessage(content=prompt),
        ]
        result = llm.invoke(messages)
        raw = result.content
        return {
            "raw": raw,
            "code": extract_code(raw),
            "explanation": extract_explanation(raw),
            "iterations": state["iterations"] + 1,
        }

    def check(state: CodeGenState) -> dict:
        # Safe by default: only execute when explicitly enabled and python.
        if not state["run_code_check"] or state["language"] != "python":
            return {"tests_passed": None}
        return {"tests_passed": executor(state["code"], timeout)}

    def route_after_check(state: CodeGenState) -> str:
        passed = state["tests_passed"]
        if passed is None or passed:
            return END
        if state["iterations"] >= state["max_iters"]:
            return END
        return "generate"

    graph = StateGraph(CodeGenState)
    graph.add_node("generate", generate)
    graph.add_node("check", check)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "check")
    graph.add_conditional_edges(
        "check", route_after_check, {"generate": "generate", END: END}
    )
    return graph.compile()


def run_codegen(
    graph,
    task: str,
    *,
    language: str,
    run_code_check: bool = False,
    max_iters: int = 2,
) -> dict:
    """Invoke the compiled graph and shape the response dict."""
    out = graph.invoke(
        {
            "task": task,
            "language": language,
            "raw": "",
            "code": "",
            "explanation": "",
            "tests_passed": None,
            "run_code_check": run_code_check,
            "iterations": 0,
            "max_iters": max_iters,
        }
    )
    return {
        "code": out["code"],
        "language": language,
        "explanation": out["explanation"],
        "tests_passed": out["tests_passed"],
    }
