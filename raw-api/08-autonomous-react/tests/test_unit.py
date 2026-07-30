# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""Unit tests for UC8 autonomous-react (raw-api) — fully mocked, no network.

We script a sequence of LLM replies (a scripted ``llm_call``) so the ReAct loop
runs offline and deterministically. The scripts force BOTH tools, exercise the
unsafe-calculator rejection, and cover both ``stopped_reason`` outcomes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import json

from app import llm, react, tools
from app import trace as trace_mod
from app.main import create_app
from app.settings import Settings


class ScriptedLLM:
    """Returns queued replies in order, recording the messages it was sent.

    When the queue is exhausted it repeats the last reply, so a one-element
    script that emits an Action will loop, and a garbled script stays garbled.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        if len(self._replies) > 1:
            return self._replies.pop(0)
        return self._replies[0]


# --------------------------------------------------------------------------- #
# tools: calculator safety
# --------------------------------------------------------------------------- #
def test_calculator_evaluates_arithmetic():
    assert tools.calculator("30 * 2") == "60"
    assert tools.calculator("99 + 1") == "100"
    assert tools.calculator("(2 + 3) * 4") == "20"


def test_calculator_rejects_unsafe_input():
    import pytest

    for hostile in ["__import__('os')", "open('x')", "a + b", "1; 2"]:
        with pytest.raises(tools.UnsafeExpression):
            tools.safe_eval(hostile)


def test_search_finds_fact():
    out = tools.search("return window")
    assert "30 days" in out


def test_search_no_match():
    assert "No matching fact" in tools.search("zzz nonexistent qqq")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_action_and_final():
    text = "Thought: I should search\nAction: search\nAction Input: return window"
    assert react.parse_action(text) == ("search", "return window")
    assert react.parse_final_answer(text) is None

    final = "Thought: done\nFinal Answer: 60"
    assert react.parse_final_answer(final) == "60"


# --------------------------------------------------------------------------- #
# loop: forces BOTH tools then finishes
# --------------------------------------------------------------------------- #
def test_loop_drives_both_tools_then_final_answer():
    script = ScriptedLLM([
        "Thought: look up the window\nAction: search\nAction Input: return window",
        "Thought: double it\nAction: calculator\nAction Input: 30*2",
        "Thought: I have the answer\nFinal Answer: 60",
    ])

    result = react.run_react("return window doubled?", llm_call=script, max_steps=6)

    assert result.stopped_reason == "final_answer"
    assert result.answer == "60"
    assert [s.action for s in result.steps] == ["search", "calculator"]
    # Parsed inputs were threaded to the tools and observations captured.
    assert result.steps[0].action_input == "return window"
    assert "30 days" in result.steps[0].observation
    assert result.steps[1].action_input == "30*2"
    assert result.steps[1].observation == "60"


def test_loop_unsafe_calculator_input_is_rejected_in_observation():
    script = ScriptedLLM([
        "Thought: try something nasty\nAction: calculator\nAction Input: __import__('os')",
        "Thought: ok\nFinal Answer: blocked",
    ])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert "unsafe" in result.steps[0].observation.lower()
    assert result.stopped_reason == "final_answer"


def test_loop_hits_max_steps_when_never_finishing():
    # Always emits an Action, never a Final Answer.
    never_done = ScriptedLLM([
        "Thought: keep going\nAction: search\nAction Input: warranty"
    ] * 10)
    result = react.run_react("loop forever", llm_call=never_done, max_steps=3)
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


def test_loop_nudges_once_then_stops_on_garbled_output():
    script = ScriptedLLM(["I have no idea what format to use."])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert result.stopped_reason == "max_steps"
    # The nudge message was appended before the second (empty-script) call.
    assert len(script.calls) == 2


def test_unknown_tool_surfaces_error_observation():
    script = ScriptedLLM([
        "Thought: bad tool\nAction: teleport\nAction Input: moon",
        "Thought: done\nFinal Answer: nope",
    ])
    result = react.run_react("x", llm_call=script, max_steps=6)
    assert "unknown tool" in result.steps[0].observation.lower()


# --------------------------------------------------------------------------- #
# HTTP level (mocked openai client, no network)
# --------------------------------------------------------------------------- #
def _fake_openai_client(replies: list[str]) -> MagicMock:
    client = MagicMock()
    queue = list(replies)

    def _create(**kwargs):
        text = queue.pop(0) if queue else "Final Answer: done"
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        # Integer usage on purpose: a bare MagicMock would sail through as a
        # "token count" and only blow up later at JSON serialisation.
        return MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=100, completion_tokens=20),
        )

    client.chat.completions.create.side_effect = _create
    return client


def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client([
        "Thought: search\nAction: search\nAction Input: return window",
        "Thought: double\nAction: calculator\nAction Input: 30*2",
        "Thought: done\nFinal Answer: 60",
    ])

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {
        "status": "ok",
        "approach": "raw-api",
        "usecase": "08-autonomous-react",
    }

    r = client.post("/run", json={"task": "return window doubled?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "60"
    assert body["stopped_reason"] == "final_answer"
    assert [s["action"] for s in body["steps"]] == ["search", "calculator"]


# --------------------------------------------------------------------------- #
# Stop-sequence handling (the gateway's Anthropic path 500s on `stop`)
# --------------------------------------------------------------------------- #
def test_supports_stop_is_false_only_for_claude_models():
    assert llm.model_profile("qwen-local-coder")["supports_stop"] is True
    assert llm.model_profile("qwen3:1.7b")["supports_stop"] is True
    assert llm.model_profile("claude-haiku")["supports_stop"] is False


def test_truncate_at_stop_cuts_hallucinated_observation():
    text = "Thought: compute\nAction: calculator\nObservation: fabricated!"
    cut = llm.truncate_at_stop(text, ["Observation:"])
    assert cut == "Thought: compute\nAction: calculator"
    assert llm.truncate_at_stop("Thought: done  ", ["Observation:"]) == "Thought: done"


def test_chat_omits_stop_for_claude_but_still_truncates():
    """Regression: sending `stop` to claude-* 500s, so the cut happens locally."""
    client = _fake_openai_client(["Action: calculator\nObservation: fake"])
    out = llm.chat(
        client,
        model="claude-haiku",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        stop=["Observation:"],
    )
    assert "stop" not in client.chat.completions.create.call_args.kwargs
    assert out == "Action: calculator"


def test_chat_sends_stop_for_models_that_support_it():
    client = _fake_openai_client(["Action: calculator"])
    llm.chat(
        client,
        model="qwen-local-coder",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        stop=["Observation:"],
    )
    assert client.chat.completions.create.call_args.kwargs["stop"] == ["Observation:"]


# --------------------------------------------------------------------------- #
# Tracing (schema + privacy). See docs/trace-format.md
# --------------------------------------------------------------------------- #
def _traced_run(**settings_kw):
    app = create_app()
    app.state.settings = Settings(**settings_kw)
    app.state.client = _fake_openai_client([
        "Thought: search\nAction: search\nAction Input: return window",
        "Thought: done\nFinal Answer: 30 days",
    ])
    return TestClient(app).post(
        "/run?trace=1", json={"task": "return window?"}
    ).json()


def test_trace_absent_unless_requested():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(["Thought: done\nFinal Answer: 30"])
    body = TestClient(app).post("/run", json={"task": "x"}).json()
    assert body["trace"] is None


def test_trace_records_llm_and_tool_spans_in_order():
    trace = _traced_run()["trace"]

    assert trace["schema_version"] == 1
    assert trace["approach"] == "raw-api"
    assert trace["usecase"] == "08-autonomous-react"
    # OTel GenAI naming, so this can be exported to Langfuse/Phoenix later.
    assert trace["gen_ai"]["system"] == "openai"
    assert trace["gen_ai"]["request"]["model"]

    kinds = [(s["seq"], s["type"], s["name"]) for s in trace["spans"]]
    assert kinds == [(1, "llm", "chat"), (2, "tool", "search"), (3, "llm", "chat")]

    assert trace["outcome"]["stop_reason"] == "final_answer"
    assert trace["outcome"]["steps"] == 2
    assert trace["outcome"]["tool_calls"] == 1
    # Unpriced endpoint: unknown, not free.
    assert trace["outcome"]["cost_usd"] is None
    # Usage totals roll up across the run's model calls (2 x 100/20).
    assert trace["gen_ai"]["usage"] == {"input_tokens": 200, "output_tokens": 40}


def test_trace_shows_exactly_what_was_sent_to_the_model():
    """The point of the raw-api approach: every byte is visible."""
    trace = _traced_run()["trace"]
    first_llm = next(s for s in trace["spans"] if s["type"] == "llm")

    roles = [m["role"] for m in first_llm["request"]["messages"]]
    assert roles[:2] == ["system", "user"]
    assert "Task: return window?" in first_llm["request"]["messages"][1]["content"]
    assert "Action: search" in first_llm["response"]["content"]


def test_trace_reflects_whether_stop_was_actually_sent():
    """The trace must show the wire truth, not the intent.

    `stop` is withheld from claude-* because the gateway 500s on it, so a trace
    of a claude run must NOT claim a stop sequence was sent — that discrepancy
    is exactly the kind of thing a trace exists to reveal.
    """
    claude = _traced_run(llm_model="claude-haiku")["trace"]
    assert "stop" not in claude["spans"][0]["request"]

    qwen = _traced_run(llm_model="qwen-local-coder")["trace"]
    assert qwen["spans"][0]["request"]["stop"] == ["Observation:"]


def test_trace_can_omit_prompt_bodies():
    """Traces carry the caller's input, so content must be suppressible."""
    trace = _traced_run(trace_include_prompts=False)["trace"]

    for span in trace["spans"]:
        body = span["request"].get("messages", span["request"].get("input"))
        assert body == trace_mod.REDACTED, span
        assert span["response"]["content"] == trace_mod.REDACTED
    # Metadata survives redaction — that is what makes it still useful.
    assert trace["outcome"]["tool_calls"] == 1
    assert trace["spans"][0]["duration_ms"] >= 0


def test_summarise_is_one_flat_row_per_run():
    """The shape that aggregates across approaches and, later, frameworks."""
    row = trace_mod.summarise(_traced_run()["trace"])
    assert set(row) == {
        "run_id", "ts", "approach", "usecase", "model", "status", "stop_reason",
        "steps", "tool_calls", "input_tokens", "output_tokens", "cost_usd",
        "duration_ms",
    }
    assert row["approach"] == "raw-api"
    assert row["stop_reason"] == "final_answer"


def test_file_sink_writes_run_and_appends_index(tmp_path):
    app = create_app()
    app.state.settings = Settings(trace_sink="file", trace_dir=str(tmp_path))
    app.state.client = _fake_openai_client(["Thought: done\nFinal Answer: 30"])

    body = TestClient(app).post("/run", json={"task": "x"}).json()
    # Sink is independent of `?trace=1`: persisted, not returned.
    assert body["trace"] is None

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1

    rows = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["run_id"] == doc["run_id"]


def test_strip_thinking_removes_qwen3_think_blocks():
    """`/no_think` still emits an empty <think></think> pair; it must not ship.

    Found by running the Docker quickstart, whose default model is a qwen3 tag:
    answers came back with a leading empty thinking block before the text.
    """
    from app.llm import strip_thinking

    empty_block = "<think>" + "\n\n" + "</think>" + "\n\n" + "30 days."
    assert strip_thinking(empty_block) == "30 days."
    assert strip_thinking("<think>reasoning</think>Answer.") == "Answer."
    assert strip_thinking("  30 days.  ") == "30 days."
    # Chain-of-thought must never survive into a response.
    assert "reasoning" not in strip_thinking("<think>reasoning</think>ok")


# --------------------------------------------------------------------------- #
# Streaming (SSE). See docs/streaming.md
# --------------------------------------------------------------------------- #
def test_think_filter_drops_reasoning_split_across_chunks():
    """The hard part of streaming: tags arrive split, and once you have
    forwarded reasoning to the client you cannot take it back."""
    f = llm.ThinkFilter()
    out = "".join(f.feed(c) for c in ["<th", "ink>secret", " stuff</thi", "nk>Answer."])
    out += f.flush()
    assert out == "Answer."
    assert "secret" not in out


def test_think_filter_passes_plain_text_through():
    f = llm.ThinkFilter()
    assert "".join(f.feed(c) for c in ["Hello ", "world"]) + f.flush() == "Hello world"


def test_think_filter_holds_back_a_possible_partial_tag():
    """A trailing '<' might begin a think tag, so it must not be emitted yet."""
    f = llm.ThinkFilter()
    assert f.feed("done <") == "done "
    assert f.flush() == "<"


def test_iter_react_yields_tokens_steps_then_final():
    """One loop, two APIs: the streaming path emits the same decisions."""
    scripted = [
        ["Thought: look", " it up", "\nAction: search\nArguments: return window"],
        ["Thought: done", "\nFinal Answer: 30 days"],
    ]
    calls = {"n": 0}

    def fake_stream(_messages):
        chunks = scripted[min(calls["n"], len(scripted) - 1)]
        calls["n"] += 1
        yield from chunks

    events = list(react.iter_react("x", llm_stream=fake_stream, max_steps=4))
    kinds = [e["type"] for e in events]

    assert kinds.count("token") == 5
    assert "step" in kinds and kinds[-1] == "final"
    # Tool ran between the two model turns.
    assert kinds.index("step") < kinds.index("final")

    step = next(e["step"] for e in events if e["type"] == "step")
    assert step.action == "search"
    assert "30 days" in step.observation

    result = events[-1]["result"]
    assert result.answer == "30 days"
    assert result.stopped_reason == "final_answer"


def test_run_react_and_iter_react_agree():
    """run_react drains iter_react, so they must not diverge."""
    script = ScriptedLLM([
        "Thought: look\nAction: search\nArguments: return window",
        "Thought: done\nFinal Answer: 30 days",
    ])
    blocking = react.run_react("x", llm_call=script, max_steps=4)

    script2 = ScriptedLLM([
        "Thought: look\nAction: search\nArguments: return window",
        "Thought: done\nFinal Answer: 30 days",
    ])
    streamed = [
        e for e in react.iter_react("x", llm_call=script2, max_steps=4)
    ][-1]["result"]

    assert blocking.answer == streamed.answer
    assert blocking.stopped_reason == streamed.stopped_reason
    assert [s.action for s in blocking.steps] == [s.action for s in streamed.steps]


def test_stream_endpoint_emits_sse_frames():
    app = create_app()
    app.state.settings = Settings()

    def fake_create(**kwargs):
        assert kwargs.get("stream") is True, "the endpoint must ask for a stream"
        for piece in ["Thought: done", "\nFinal Answer: 30 days"]:
            delta = MagicMock()
            delta.content = piece
            choice = MagicMock()
            choice.delta = delta
            yield MagicMock(choices=[choice])

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    app.state.client = client

    with TestClient(app).stream(
        "POST", "/run/stream", json={"task": "return window?"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    assert "event: token" in body
    assert "event: final" in body
    # Frames are terminated by a blank line, or the client buffers forever.
    assert body.endswith("\n\n")
    final = json.loads(body.rsplit("data: ", 1)[1].strip())
    assert final["answer"] == "30 days"
    assert final["stopped_reason"] == "final_answer"
