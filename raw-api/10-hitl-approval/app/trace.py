# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC10 HITL approval (raw-api). See docs/trace-format.md
"""Trace recording: what the agent actually did, in a portable shape.

Field names follow the OpenTelemetry GenAI semantic conventions so these traces
can later be exported to Langfuse / Phoenix / Jaeger with a small adapter — but
nothing here depends on a tracing library, a server, or a key. It is stdlib
JSON, because every example in this repo must stay clonable and air-gap-runnable.

See ``docs/trace-format.md`` for the schema and the reasoning.
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

# This approach talks the OpenAI wire protocol, whatever model sits behind it.
GEN_AI_SYSTEM = "openai"

REDACTED = "<omitted: TRACE_INCLUDE_PROMPTS=0>"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Tracer:
    """Accumulates spans for a single ``/run``.

    Deliberately dumb: no context vars, no global state, no background flushing.
    One tracer per request, passed explicitly, so the data flow stays visible —
    which is the whole point of the raw-api approach.
    """

    approach: str
    usecase: str
    model: str
    temperature: float
    max_tokens: int
    include_prompts: bool = True

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=_now_iso)
    _t0: float = field(default_factory=time.perf_counter)
    spans: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def _body(self, value: Any) -> Any:
        return value if self.include_prompts else REDACTED

    def llm_span(
        self,
        *,
        messages: list[dict],
        completion: str,
        duration_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop: list[str] | None = None,
    ) -> None:
        """Record one model call, including the exact messages sent."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        request: dict[str, Any] = {"messages": self._body(messages)}
        if stop:
            request["stop"] = stop
        self.spans.append(
            {
                "seq": len(self.spans) + 1,
                "type": "llm",
                "name": "chat",
                "duration_ms": duration_ms,
                "gen_ai": {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
                "request": request,
                "response": {"content": self._body(completion)},
            }
        )

    def tool_span(
        self, *, name: str, tool_input: str, output: str, duration_ms: int
    ) -> None:
        """Record one tool invocation and what it returned."""
        self.spans.append(
            {
                "seq": len(self.spans) + 1,
                "type": "tool",
                "name": name,
                "duration_ms": duration_ms,
                "request": {"input": self._body(tool_input)},
                "response": {"content": self._body(output)},
            }
        )

    def finish(self, *, status: str, stop_reason: str) -> dict:
        """Close the run and return the trace document."""
        tool_calls = sum(1 for s in self.spans if s["type"] == "tool")
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "duration_ms": int((time.perf_counter() - self._t0) * 1000),
            "approach": self.approach,
            "usecase": self.usecase,
            "gen_ai": {
                "system": GEN_AI_SYSTEM,
                "request": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            },
            "outcome": {
                "status": status,
                "stop_reason": stop_reason,
                "steps": sum(1 for s in self.spans if s["type"] == "llm"),
                "tool_calls": tool_calls,
                # An OpenAI-compatible endpoint reports tokens, not price. null
                # rather than 0.0: an unpriced run is unknown, not free.
                "cost_usd": None,
            },
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
