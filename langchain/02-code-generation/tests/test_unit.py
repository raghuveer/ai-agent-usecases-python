# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC2 Code generation (langchain). See langchain/02-code-generation/README.md
"""Unit tests for UC2 code-generation (langchain) — fully mocked, no network.

We inject ``FakeListChatModel`` onto ``app.state`` and exercise the LCEL chain
directly. Tests also assert the safety contract: flag off => no code executes.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import codegen
from app.codegen import _system_prompt, build_chain, extract_code, extract_explanation
from app.main import app

FENCED_REPLY = (
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "```\n"
    "This defines add, which returns the sum of its two arguments."
)


def make_client(responses: list[str]) -> TestClient:
    app.state.llm = FakeListChatModel(responses=responses)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Pure-function level
# --------------------------------------------------------------------------- #
def test_extract_code_pulls_first_fenced_block():
    code = extract_code(FENCED_REPLY)
    assert "def add(a, b):" in code
    assert "```" not in code


def test_extract_code_no_fence_falls_back():
    assert extract_code("  raw code  ") == "raw code"


def test_extract_explanation_is_prose_after_block():
    expl = extract_explanation(FENCED_REPLY)
    assert expl.startswith("This defines add")


def test_no_think_prefix_for_qwen3():
    assert _system_prompt("qwen3:1.7b").startswith("/no_think")
    assert not _system_prompt("qwen-local-coder").startswith("/no_think")


def test_chain_returns_model_text():
    llm = FakeListChatModel(responses=[FENCED_REPLY])
    chain = build_chain(llm, "qwen-local-coder")
    out = chain.invoke({"task": "add", "language": "python"})
    assert out == FENCED_REPLY


def test_generate_extracts_and_does_not_execute_when_flag_off(monkeypatch):
    spy = {"called": False}

    def _spy(code, timeout):  # pragma: no cover - must not run
        spy["called"] = True
        return True

    monkeypatch.setattr(codegen, "smoke_check_python", _spy)
    llm = FakeListChatModel(responses=[FENCED_REPLY])
    result = codegen.generate(
        "Write add(a,b)",
        language="python",
        llm=llm,
        model_name="qwen-local-coder",
        run_code_check=False,
    )
    assert spy["called"] is False
    assert result["tests_passed"] is None
    assert "def add(a, b):" in result["code"]
    assert result["language"] == "python"
    assert result["explanation"].startswith("This defines add")


def test_generate_runs_check_only_when_flag_on(monkeypatch):
    monkeypatch.setattr(codegen, "smoke_check_python", lambda c, t: True)
    llm = FakeListChatModel(responses=[FENCED_REPLY])
    result = codegen.generate(
        "Write add(a,b)",
        language="python",
        llm=llm,
        model_name="qwen-local-coder",
        run_code_check=True,
    )
    assert result["tests_passed"] is True


def test_generate_iterates_on_failure_capped_at_two(monkeypatch):
    attempts = {"n": 0}

    def always_fail(code, timeout):
        attempts["n"] += 1
        return False

    monkeypatch.setattr(codegen, "smoke_check_python", always_fail)
    # Two responses available for initial + one fix attempt.
    llm = FakeListChatModel(responses=[FENCED_REPLY, FENCED_REPLY])
    result = codegen.generate(
        "Write add(a,b)",
        language="python",
        llm=llm,
        model_name="qwen-local-coder",
        run_code_check=True,
        max_iters=2,
    )
    assert result["tests_passed"] is False
    assert attempts["n"] == 2  # capped


def test_check_skipped_for_non_python(monkeypatch):
    spy = {"called": False}

    def _spy(code, timeout):  # pragma: no cover
        spy["called"] = True
        return True

    monkeypatch.setattr(codegen, "smoke_check_python", _spy)
    llm = FakeListChatModel(responses=["```js\nconst f=()=>1;\n```\nok"])
    result = codegen.generate(
        "make f",
        language="javascript",
        llm=llm,
        model_name="qwen-local-coder",
        run_code_check=True,
    )
    assert spy["called"] is False
    assert result["tests_passed"] is None


# --------------------------------------------------------------------------- #
# HTTP level (fake LLM, no network)
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client(["unused"])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langchain",
        "usecase": "02-code-generation",
    }


def test_run_returns_code_and_explanation():
    client = make_client([FENCED_REPLY])
    resp = client.post("/run", json={"task": "Write add(a,b)"})
    assert resp.status_code == 200
    body = resp.json()
    assert "def add(a, b):" in body["code"]
    assert body["language"] == "python"  # default
    assert body["explanation"].startswith("This defines add")
    assert body["tests_passed"] is None  # flag off => not executed


def test_run_respects_language_override():
    client = make_client(["```js\nconst f = () => 1;\n```\nok"])
    resp = client.post("/run", json={"task": "make f", "language": "javascript"})
    assert resp.status_code == 200
    assert resp.json()["language"] == "javascript"
