# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC7 Multi-agent (claude-agent-sdk). See docs/trace-format.md
"""Trace recording: what the agent actually did, in a portable shape.

Field names follow the OpenTelemetry GenAI semantic conventions so these traces
can later be exported to Langfuse / Phoenix / Jaeger with a small adapter — but
nothing here depends on a tracing library, a server, or a key.

**The agent-SDK difference, and it is the point of this file.** The other three
approaches own their loop, so they can record the exact message list of every
model call, each tool's *result*, and per-call latency. Here the SDK owns the
loop: it reports which tools were called and with what, plus run totals — and
nothing else. So this trace is honestly partial:

* no per-call message lists — the harness builds each request, not this code;
* no tool *results* — the harness runs the tool and feeds the output back
  internally;
* no token counts — the SDK reports cost, not usage;
* no per-call latency — only whole-run duration is observable.

Those gaps are listed in ``not_captured`` rather than filled with zeros. A zero
token count would read as "this run used no tokens", which is false; absent data
must look absent. The asymmetry *is* the finding — writing no loop costs you the
ability to see inside it.

What this approach knows that the others cannot: a real ``cost_usd``.

See ``docs/trace-format.md`` for the shared schema.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# The SDK speaks the Anthropic protocol (the CLI appends /v1/messages).
GEN_AI_SYSTEM = "anthropic"

REDACTED = "<omitted: TRACE_INCLUDE_PROMPTS=0>"

# Recorded on every trace from this approach so a reader comparing it with
# raw-api / langchain / langgraph knows the gaps are structural, not a bug here.
NOT_CAPTURED = [
    "request.messages: the SDK builds each request inside the harness",
    "tool results: the harness executes tools and feeds output back internally",
    "gen_ai.usage: the SDK reports cost, not token counts",
    "per-call latency: only whole-run duration is observable",
]


# The trace vocabulary is deliberately cross-approach (see docs/trace-format.md):
# raw-api / langchain / langgraph own their loops and report `final_answer` or
# `max_steps`. The SDK speaks Anthropic's vocabulary instead -- `end_turn`,
# `max_tokens` -- so it is translated here rather than leaked into the trace.
# Leaking it would silently make this approach's runs incomparable with the
# other three, which is the one thing the shared format exists to prevent.
_TRACE_REASONS = {
    "end_turn": "final_answer",
    "stop_sequence": "final_answer",
    "tool_use": "final_answer",
}


def to_trace_reason(sdk_reason: str) -> str:
    """Map an SDK stop reason onto the shared trace vocabulary.

    Unknown reasons pass through unchanged: `max_turns`, `max_budget`,
    `max_tokens` and `error` already mean the same thing in both vocabularies.
    """
    return _TRACE_REASONS.get(sdk_reason, sdk_reason)


def status_for(trace_reason: str) -> str:
    """The `status` implied by a stop reason: ok | capped | error."""
    if trace_reason == "final_answer":
        return "ok"
    return "capped" if trace_reason.startswith("max_") else "error"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Tracer:
    """Builds a trace from an ``AgentResult`` — after the fact, not during it.

    There is no streaming hook to attach: ``collect()`` folds the SDK's message
    stream and this reads whatever survived. Another consequence of not owning
    the loop.
    """

    approach: str
    usecase: str
    model: str
    max_turns: int
    max_budget_usd: float
    include_prompts: bool = True

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=_now_iso)
    _t0: float = field(default_factory=time.perf_counter)
    spans: list[dict] = field(default_factory=list)

    def _body(self, value: Any) -> Any:
        return value if self.include_prompts else REDACTED

    def tool_span(self, *, name: str, tool_input: dict) -> None:
        """Record a tool call. ``response`` is null — the harness kept the result."""
        self.spans.append(
            {
                "seq": len(self.spans) + 1,
                "type": "tool",
                "name": name,
                "duration_ms": None,  # unobservable from outside the harness
                "request": {"input": self._body(tool_input)},
                "response": None,
            }
        )

    def finish(
        self,
        *,
        status: str,
        stop_reason: str,
        num_turns: int,
        cost_usd: float | None,
        duration_ms: int | None = None,
    ) -> dict:
        """``duration_ms`` must be passed in.

        This tracer is built *after* the run, from the finished ``AgentResult``,
        so its own clock started too late to mean anything — the caller times the
        run and hands the number over. Whole-run duration is the one timing this
        approach can honestly report.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "duration_ms": duration_ms,
            "approach": self.approach,
            "usecase": self.usecase,
            "gen_ai": {
                "system": GEN_AI_SYSTEM,
                "request": {
                    "model": self.model,
                    "max_turns": self.max_turns,
                    "max_budget_usd": self.max_budget_usd,
                },
                # null, not 0: the SDK does not report token usage, and a zero
                # here would be read as a measurement.
                "usage": {"input_tokens": None, "output_tokens": None},
            },
            "outcome": {
                "status": status,
                "stop_reason": stop_reason,
                # Turns, not model calls: the harness may make several calls per
                # turn and does not break them out.
                "steps": num_turns,
                "tool_calls": len(self.spans),
                # The one number this approach knows better than the others.
                "cost_usd": cost_usd,
            },
            "not_captured": NOT_CAPTURED,
            "spans": self.spans,
        }


def summarise(trace: dict) -> dict:
    """One flat row per run — the shape that aggregates across frameworks."""
    gen_ai, outcome = trace["gen_ai"], trace["outcome"]
    return {
        "run_id": trace["run_id"],
        "ts": trace["started_at"],
        "approach": trace["approach"],
        "usecase": trace["usecase"],
        "model": gen_ai["request"]["model"],
        "status": outcome["status"],
        "stop_reason": outcome["stop_reason"],
        "steps": outcome["steps"],
        "tool_calls": outcome["tool_calls"],
        "input_tokens": gen_ai["usage"]["input_tokens"],
        "output_tokens": gen_ai["usage"]["output_tokens"],
        "cost_usd": outcome["cost_usd"],
        "duration_ms": trace["duration_ms"],
    }


def persist(trace: dict, trace_dir: str) -> Path:
    """Write ``<trace_dir>/<run_id>.json`` and append to ``runs.jsonl``."""
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace['run_id']}.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    with (directory / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summarise(trace)) + "\n")
    return path
