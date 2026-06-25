"""Unit tests for UC2 code-generation (langgraph). Fake LLM, no network.

Tests also assert the safety contract: with the self-check flag OFF, the check
node never calls the executor (no generated code runs).
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.codegen import (
    build_codegen_graph,
    extract_code,
    extract_explanation,
    run_codegen,
)
from app.main import create_app

FENCED_REPLY = (
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "```\n"
    "This defines add, which returns the sum of its two arguments."
)


def make_client(answer: str = FENCED_REPLY) -> TestClient:
    llm = FakeListChatModel(responses=[answer])
    return TestClient(create_app(llm=llm))


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
    assert extract_explanation(FENCED_REPLY).startswith("This defines add")


# --------------------------------------------------------------------------- #
# Graph level
# --------------------------------------------------------------------------- #
def test_graph_extracts_and_does_not_execute_when_flag_off():
    called = {"exec": False}

    def fake_exec(code, timeout):  # pragma: no cover - must not run
        called["exec"] = True
        return True

    llm = FakeListChatModel(responses=[FENCED_REPLY])
    graph = build_codegen_graph(llm, executor=fake_exec)
    result = run_codegen(graph, "Write add(a,b)", language="python", run_code_check=False)

    assert called["exec"] is False
    assert result["tests_passed"] is None
    assert "def add(a, b):" in result["code"]
    assert result["language"] == "python"
    assert result["explanation"].startswith("This defines add")


def test_graph_runs_check_only_when_flag_on():
    llm = FakeListChatModel(responses=[FENCED_REPLY])
    graph = build_codegen_graph(llm, executor=lambda code, timeout: True)
    result = run_codegen(graph, "Write add(a,b)", language="python", run_code_check=True)
    assert result["tests_passed"] is True


def test_graph_iterates_on_failure_capped_at_two():
    attempts = {"n": 0}

    def always_fail(code, timeout):
        attempts["n"] += 1
        return False

    # Provide enough responses for the initial gen + one fix attempt.
    llm = FakeListChatModel(responses=[FENCED_REPLY, FENCED_REPLY, FENCED_REPLY])
    graph = build_codegen_graph(llm, executor=always_fail)
    result = run_codegen(
        graph, "Write add(a,b)", language="python", run_code_check=True, max_iters=2
    )
    assert result["tests_passed"] is False
    assert attempts["n"] == 2  # check ran twice (capped)


def test_graph_skips_check_for_non_python():
    called = {"exec": False}

    def fake_exec(code, timeout):  # pragma: no cover
        called["exec"] = True
        return True

    llm = FakeListChatModel(responses=["```js\nconst f=()=>1;\n```\nok"])
    graph = build_codegen_graph(llm, executor=fake_exec)
    result = run_codegen(graph, "make f", language="javascript", run_code_check=True)
    assert called["exec"] is False
    assert result["tests_passed"] is None


# --------------------------------------------------------------------------- #
# HTTP level
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "02-code-generation",
    }


def test_run_returns_code_and_explanation():
    client = make_client()
    resp = client.post("/run", json={"task": "Write add(a,b)"})
    assert resp.status_code == 200
    body = resp.json()
    assert "def add(a, b):" in body["code"]
    assert body["language"] == "python"  # default
    assert body["explanation"].startswith("This defines add")
    assert body["tests_passed"] is None  # flag off => not executed


def test_run_respects_language_override():
    client = make_client("```js\nconst f = () => 1;\n```\nok")
    resp = client.post("/run", json={"task": "make f", "language": "javascript"})
    assert resp.status_code == 200
    assert resp.json()["language"] == "javascript"
