#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
"""Run one use case live through all four approaches and record the comparison.

Writes `docs/compare/data/<usecase>.json` — run summaries only, never prompts or
completions, so the committed data carries no request content. The generated
comparison page embeds it, which means the page can be rebuilt offline and the
numbers are refreshed deliberately rather than on every render.

    python scripts/trace_compare.py 08-autonomous-react

Costs real money: it drives each approach against the gateway, and the
claude-agent-sdk project additionally needs Node + the Claude Code CLI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "docs" / "compare" / "data"

# Each approach exposes /run slightly differently; the driver is per-approach on
# purpose rather than pretending one shape fits all.
TASKS: dict[str, dict[str, object]] = {
    "08-autonomous-react": {
        "task": "What is the Northwind return window, doubled?",
        "question": "What is the revenue decline, as a percentage?",
    },
}

# Executed inside each project with its own venv, so every approach runs against
# exactly the dependencies it pins.
DRIVER = r'''
import json, sys
from fastapi.testclient import TestClient
approach = sys.argv[1]
payload = json.loads(sys.argv[2])
if approach == "langchain":
    from app.main import app
    from app.llm import build_llm
    app.state.llm = build_llm()
    make = lambda: TestClient(app)
else:
    from app.main import create_app
    make = lambda: TestClient(create_app())
with make() as client:
    body = client.post("/run?trace=1", json=payload).json()
# `run_trace` first: claude-agent-sdk also has a `trace` key holding its
# tool-call LIST, which is truthy and would win the fallback.
doc = body.get("run_trace") or body.get("trace")
if not doc:
    print(json.dumps({"error": "no trace in response", "keys": list(body)}))
    sys.exit(1)
gen_ai, outcome = doc["gen_ai"], doc["outcome"]
print(json.dumps({
    "steps": outcome["steps"],
    "tool_calls": outcome["tool_calls"],
    "input_tokens": gen_ai["usage"]["input_tokens"],
    "output_tokens": gen_ai["usage"]["output_tokens"],
    "cost_usd": outcome["cost_usd"],
    "duration_ms": doc["duration_ms"],
    "stop_reason": outcome["stop_reason"],
    "graph_path": doc.get("graph_path"),
    "not_captured": doc.get("not_captured"),
}))
'''


def run_one(approach: str, usecase: str, payload: dict) -> dict:
    project = REPO / approach / usecase
    python = project / ".venv" / "Scripts" / "python.exe"
    if not python.exists():  # POSIX layout
        python = project / ".venv" / "bin" / "python"
    if not python.exists():
        return {"error": "no .venv — run `uv sync --extra dev` in the project"}

    proc = subprocess.run(
        [str(python), "-c", DRIVER, approach, json.dumps(payload)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return {"error": " / ".join(tail) or f"exit {proc.returncode}"}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usecase")
    args = parser.parse_args()
    if args.usecase not in TASKS:
        raise SystemExit(f"no task defined for {args.usecase!r}")

    spec = TASKS[args.usecase]
    runs: dict[str, dict] = {}
    model = ""
    for approach in ("raw-api", "langchain", "langgraph", "claude-agent-sdk"):
        # claude-agent-sdk's /run takes `question`; the other three take `task`.
        payload = (
            {"question": spec["question"]}
            if approach == "claude-agent-sdk"
            else {"task": spec["task"]}
        )
        print(f"running {approach}/{args.usecase} …", flush=True)
        result = run_one(approach, args.usecase, payload)
        runs[approach] = result
        print(f"  {result}", flush=True)

    notes = [
        "`raw-api`, `langchain` and `langgraph` send identical payloads — the "
        "frameworks add no prompt overhead for this use case.",
        "`claude-agent-sdk` reports cost but not tokens; the other three report "
        "tokens but not cost, because an OpenAI-compatible endpoint does not "
        "price the call. Unknown is recorded as null, never 0.",
        "The agent-SDK task is phrased for its own tool set, so cost is not a "
        "controlled head-to-head — the order of magnitude is the point.",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{args.usecase}.json"
    out.write_text(
        json.dumps(
            {
                "usecase": args.usecase,
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "model": model or "claude-haiku",
                "runs": runs,
                "notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {out.relative_to(REPO)}")
    print("now run: python scripts/compare_usecase.py", args.usecase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
