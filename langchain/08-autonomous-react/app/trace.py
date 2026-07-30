# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (langchain). See docs/trace-format.md
"""Trace recording: what the agent actually did, in a portable shape.

Field names follow the OpenTelemetry GenAI semantic conventions so these traces
can later be exported to Langfuse / Phoenix / Jaeger with a small adapter — but
nothing here depends on a tracing library, a server, or a key. It is stdlib
JSON, because every example in this repo must stay clonable and air-gap-runnable.

**The langchain difference.** The raw-api version instruments by hand, because it
owns the HTTP call. Here the framework already has an observation seam —
``BaseCallbackHandler`` — so tracing attaches with ``llm.with_config(callbacks=…)``
and ``Tool(callbacks=…)``, and the ReAct loop itself needs no changes at all.
That is the trade this repo exists to show: less code to write, and the exact
request is now whatever LangChain chose to build rather than something you can
read off the call site.

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

from langchain_core.callbacks import BaseCallbackHandler

SCHEMA_VERSION = 1

# This approach talks the OpenAI wire protocol, whatever model sits behind it.
GEN_AI_SYSTEM = "openai"

REDACTED = "<omitted: TRACE_INCLUDE_PROMPTS=0>"

# LangChain names message types `human`/`ai`; the traces use the OTel GenAI
# (and OpenAI) role names so a trace from this approach lines up field-for-field
# with one from raw-api. Comparability is the whole point of the format.
_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


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


class TracingCallbackHandler(BaseCallbackHandler):
    """Feeds a :class:`Tracer` from LangChain's own callback events.

    Attached with ``llm.with_config(callbacks=[handler])`` and by rebuilding the
    tools with ``callbacks=[handler]``, so neither ``react.py`` nor the tools
    know tracing exists.
    """

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self._llm_t0: float | None = None
        self._llm_messages: list[dict] = []
        self._tool_t0: float | None = None
        self._tool_name = ""
        self._tool_input = ""

    # -- model ------------------------------------------------------------- #
    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._llm_t0 = time.perf_counter()
        # messages is a list of message-lists (one per generation prompt).
        flat = messages[0] if messages else []
        self._llm_messages = [
            {"role": _ROLES.get(getattr(m, "type", ""), getattr(m, "type", "unknown")),
             "content": str(m.content)}
            for m in flat
        ]

    def on_llm_end(self, response, **kwargs) -> None:
        elapsed = self._llm_t0 or time.perf_counter()
        generation = response.generations[0][0] if response.generations else None
        text = getattr(generation, "text", "") or ""

        usage = {}
        message = getattr(generation, "message", None)
        if message is not None:
            usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            usage = (response.llm_output or {}).get("token_usage", {}) or {}

        self.tracer.llm_span(
            messages=self._llm_messages,
            completion=text,
            duration_ms=int((time.perf_counter() - elapsed) * 1000),
            input_tokens=_as_int(usage, "input_tokens", "prompt_tokens"),
            output_tokens=_as_int(usage, "output_tokens", "completion_tokens"),
        )

    # -- tools -------------------------------------------------------------- #
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._tool_t0 = time.perf_counter()
        self._tool_name = (serialized or {}).get("name", "unknown")
        self._tool_input = input_str

    def on_tool_end(self, output, **kwargs) -> None:
        started = self._tool_t0 or time.perf_counter()
        self.tracer.tool_span(
            name=self._tool_name,
            tool_input=self._tool_input,
            output=str(output),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _as_int(usage: dict, *names: str) -> int:
    """First integer among ``names``; 0 when absent or non-numeric.

    Providers disagree on the key names (``input_tokens`` vs ``prompt_tokens``)
    and some omit usage entirely — a missing count is 0, never a guess.
    """
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return 0


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
