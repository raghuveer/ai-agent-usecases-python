"""Unit tests for UC2 code-generation (raw-api) — fully mocked, no network.

These tests also assert the safety contract: with the self-check flag OFF, no
generated code is ever executed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import codegen, llm
from app.main import create_app
from app.settings import Settings

FENCED_REPLY = (
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "```\n"
    "This defines add, which returns the sum of its two arguments."
)


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# --------------------------------------------------------------------------- #
# Pure-function level
# --------------------------------------------------------------------------- #
def test_extract_code_pulls_first_fenced_block():
    code = codegen.extract_code(FENCED_REPLY)
    assert "def add(a, b):" in code
    assert "return a + b" in code
    assert "```" not in code


def test_extract_code_handles_language_tag_and_no_fence():
    assert codegen.extract_code("```\nprint(1)\n```") == "print(1)"
    # No fence -> fall back to the whole stripped text.
    assert codegen.extract_code("  raw code  ") == "raw code"


def test_extract_explanation_is_prose_after_block():
    expl = codegen.extract_explanation(FENCED_REPLY)
    assert expl.startswith("This defines add")
    assert "```" not in expl


def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("qwen-local-coder", "x") == "x"


def test_generate_extracts_and_does_not_execute_when_flag_off():
    captured = {}

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return FENCED_REPLY

    # Spy on the executor: it must NOT be called when run_code_check is off.
    called = {"smoke": False}
    original = codegen.smoke_check_python

    def _spy(code, timeout):  # pragma: no cover - should never run
        called["smoke"] = True
        return original(code, timeout)

    codegen.smoke_check_python = _spy
    try:
        result = codegen.generate(
            "Write add(a,b)",
            language="python",
            llm_call=fake_llm,
            run_code_check=False,
        )
    finally:
        codegen.smoke_check_python = original

    assert called["smoke"] is False
    assert result["tests_passed"] is None
    assert "def add(a, b):" in result["code"]
    assert result["language"] == "python"
    assert result["explanation"].startswith("This defines add")
    assert "Task: Write add(a,b)" in captured["user"]


def test_generate_runs_check_only_when_flag_on(monkeypatch):
    monkeypatch.setattr(codegen, "smoke_check_python", lambda code, timeout: True)
    result = codegen.generate(
        "Write add(a,b)",
        language="python",
        llm_call=lambda s, u: FENCED_REPLY,
        run_code_check=True,
    )
    assert result["tests_passed"] is True


def test_generate_iterates_on_failure_capped_at_two(monkeypatch):
    attempts = {"n": 0}

    def always_fail(code, timeout):
        attempts["n"] += 1
        return False

    calls = {"llm": 0}

    def fake_llm(s, u):
        calls["llm"] += 1
        return FENCED_REPLY

    monkeypatch.setattr(codegen, "smoke_check_python", always_fail)
    result = codegen.generate(
        "Write add(a,b)",
        language="python",
        llm_call=fake_llm,
        run_code_check=True,
        max_iters=2,
    )
    assert result["tests_passed"] is False
    assert attempts["n"] == 2  # capped
    assert calls["llm"] == 2  # initial + one fix attempt


def test_check_skipped_for_non_python(monkeypatch):
    spy = {"called": False}

    def _spy(code, timeout):  # pragma: no cover
        spy["called"] = True
        return True

    monkeypatch.setattr(codegen, "smoke_check_python", _spy)
    result = codegen.generate(
        "Write a function",
        language="javascript",
        llm_call=lambda s, u: "```javascript\nfunction f(){}\n```\nok",
        run_code_check=True,
    )
    assert spy["called"] is False
    assert result["tests_passed"] is None


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network, flag off => no execution)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(FENCED_REPLY)

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "02-code-generation",
    }

    r = client.post("/run", json={"task": "Write add(a,b)"})
    assert r.status_code == 200
    body = r.json()
    assert "def add(a, b):" in body["code"]
    assert body["language"] == "python"  # default language
    assert body["tests_passed"] is None  # flag off => not executed


def test_run_respects_language_override():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client("```js\nconst f = () => 1;\n```\nok")

    client = TestClient(app)
    r = client.post("/run", json={"task": "make f", "language": "javascript"})
    assert r.status_code == 200
    assert r.json()["language"] == "javascript"
