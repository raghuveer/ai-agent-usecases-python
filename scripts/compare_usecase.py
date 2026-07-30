#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
"""Generate a side-by-side comparison page for one use case.

The repo's whole thesis is that the four approaches differ. Until now a reader
had to open four folders and diff them mentally. This renders the comparison:
the *same* problem, the core of each implementation, how much code each costs,
and — when trace data is present — what each one measurably did at runtime.

Generated, not hand-written, so it cannot drift from the code it describes.

    python scripts/compare_usecase.py 08-autonomous-react
    python scripts/compare_usecase.py --all --check     # CI-style drift check
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APPROACHES = ("raw-api", "langchain", "langgraph", "claude-agent-sdk")
OUT_DIR = REPO / "docs" / "compare"
DATA_DIR = OUT_DIR / "data"

# Where the heart of each implementation lives: (module, symbol). Chosen so the
# rendered snippets answer the same question — "what drives the loop?" — rather
# than showing whatever happens to be at the top of a file.
MANIFEST: dict[str, dict[str, tuple[str, str]]] = {
    "01-rag": {
        "raw-api": ("app/rag.py", "answer"),
        "langchain": ("app/rag.py", "build_chain"),
        "langgraph": ("app/rag.py", "build_rag_graph"),
        "claude-agent-sdk": ("app/rag.py", "answer"),
    },
    "02-code-generation": {
        "raw-api": ("app/codegen.py", "generate"),
        "langchain": ("app/codegen.py", "generate"),
        "langgraph": ("app/codegen.py", "build_codegen_graph"),
        "claude-agent-sdk": ("app/codegen.py", "generate"),
    },
    "03-data-extraction": {
        "raw-api": ("app/extract.py", "extract_invoice"),
        "langchain": ("app/extract.py", "extract_invoice"),
        "langgraph": ("app/extract.py", "build_extract_graph"),
        "claude-agent-sdk": ("app/extract.py", "extract"),
    },
    "04-research-agent": {
        "raw-api": ("app/agent.py", "run_agent"),
        "langchain": ("app/agent.py", "run_agent"),
        "langgraph": ("app/agent.py", "build_agent_graph"),
        "claude-agent-sdk": ("app/research.py", "research"),
    },
    "05-support-triage": {
        "raw-api": ("app/triage.py", "triage"),
        "langchain": ("app/triage.py", "triage"),
        "langgraph": ("app/triage.py", "build_triage_graph"),
        "claude-agent-sdk": ("app/triage.py", "triage"),
    },
    "06-sql-agent": {
        "raw-api": ("app/sqlagent.py", "answer"),
        "langchain": ("app/sqlagent.py", "answer_question"),
        "langgraph": ("app/sqlagent.py", "build_sql_graph"),
        "claude-agent-sdk": ("app/sql_agent.py", "ask"),
    },
    "07-multi-agent": {
        "raw-api": ("app/agents.py", "orchestrate"),
        "langchain": ("app/agents.py", "orchestrate"),
        "langgraph": ("app/graph.py", "build_multi_agent_graph"),
        "claude-agent-sdk": ("app/team.py", "run_team"),
    },
    "08-autonomous-react": {
        "raw-api": ("app/react.py", "run_react"),
        "langchain": ("app/react.py", "run_react"),
        "langgraph": ("app/react.py", "build_react_graph"),
        "claude-agent-sdk": ("app/react_agent.py", "run_react"),
    },
    "09-recommendations": {
        "raw-api": ("app/recommend.py", "recommend"),
        "langchain": ("app/recommend.py", "recommend"),
        "langgraph": ("app/recommend.py", "build_recommend_graph"),
        "claude-agent-sdk": ("app/recommend.py", "recommend"),
    },
    "10-hitl-approval": {
        "raw-api": ("app/hitl.py", "start_run"),
        "langchain": ("app/hitl.py", "start_run"),
        "langgraph": ("app/hitl.py", "build_approval_graph"),
        "claude-agent-sdk": ("app/approval.py", "start_run"),
    },
}

TITLES: dict[str, str] = {
    "01-rag": "Q&A / RAG chatbot",
    "02-code-generation": "Code generation",
    "03-data-extraction": "Data extraction (structured output)",
    "04-research-agent": "Research agent",
    "05-support-triage": "Customer support triage",
    "06-sql-agent": "SQL / DB agent",
    "07-multi-agent": "Multi-agent orchestration",
    "08-autonomous-react": "Autonomous ReAct",
    "09-recommendations": "Personalised recommendations",
    "10-hitl-approval": "Human-in-the-loop approval",
}

# The claim each snippet is evidence for. Falls back to the generic line when a
# use case has nothing specific to say — better a true generality than a forced
# observation.
DEFAULT_TAKE: dict[str, str] = {
    "raw-api": "You write the control flow; every byte sent is visible at the call site.",
    "langchain": "Composition helpers do the plumbing; the control flow is still yours.",
    "langgraph": "State and control flow become a typed graph.",
    "claude-agent-sdk": "The SDK owns the loop; you supply tools and a prompt.",
}

TAKE: dict[str, dict[str, str]] = {
    "01-rag": {
        "raw-api": "Retrieve, build a prompt, call once. No framework needed.",
        "langchain": "The natural fit: a retriever and an LCEL chain, declared not written.",
        "langgraph": "A graph for a straight line — structural cost with no payoff here.",
        "claude-agent-sdk": "No vector store: retrieval is lexical Grep/Read, so phrasing can miss.",
    },
    "03-data-extraction": {
        "raw-api": "One call plus a validating parser — and a retry that tells the model what was wrong.",
        "langchain": "Structured output is a first-class chain concern.",
        "langgraph": "Extract → validate → repair is a genuine pipeline, so nodes earn their keep.",
        "claude-agent-sdk": "An agent harness doing a one-shot job: no loop ever runs.",
    },
    "07-multi-agent": {
        "raw-api": "You hand-roll the orchestrator, the hand-offs, and the review gate.",
        "langchain": "Chains per role, sequenced by hand — the coordination is not the framework's job.",
        "langgraph": "Roles are nodes and hand-offs are edges; the topology is the program.",
        "claude-agent-sdk": "Subagents are data: a dict of definitions, each with its own context and tools.",
    },
    "08-autonomous-react": {
        "raw-api": "You write the loop, the parser, and the stop condition.",
        "langchain": "The framework supplies tools and message types; the loop is still yours.",
        "langgraph": "The loop becomes a graph: nodes, a conditional edge, and a cycle.",
        "claude-agent-sdk": "There is no loop in the repo. The SDK owns it.",
    },
    "10-hitl-approval": {
        "raw-api": "A hand-built checkpoint store plus a /resume endpoint.",
        "langchain": "A callback workaround: the framework has no native pause.",
        "langgraph": "`interrupt()` — a durable pause the graph resumes from.",
        "claude-agent-sdk": "`can_use_tool` gates the action, but only in-process.",
    },
}


def take_for(usecase: str, approach: str) -> str:
    return TAKE.get(usecase, {}).get(approach) or DEFAULT_TAKE[approach]


def code_lines(path: Path) -> int:
    """Non-blank, non-comment lines — a fairer size measure than raw wc -l."""
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            total += 1
    return total


def app_size(project: Path) -> int:
    return sum(code_lines(p) for p in sorted((project / "app").glob("*.py")))


def extract(project: Path, module: str, symbol: str) -> str:
    """Return the source of ``symbol`` from ``module``.

    Uses the AST rather than regex or line markers, so the sources stay clean
    and a rename fails loudly here instead of silently rendering the wrong code.
    """
    path = project / module
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                segment = ast.get_source_segment(source, node)
                if segment:
                    return segment
    raise SystemExit(f"{path}: no top-level symbol named {symbol!r}")


def render(usecase: str) -> str:
    entries = MANIFEST[usecase]
    lines: list[str] = []
    add = lines.append

    add(f"# {usecase} — {TITLES[usecase]}, four ways")
    add("")
    add(
        "<!-- GENERATED by scripts/compare_usecase.py. Do not edit by hand; "
        "run the script. -->"
    )
    add("")
    add(
        "Every approach here solves an identical task with identical tools. What "
        "differs is how much of the machinery you write yourself — and, as the "
        "measured section below shows, what you can still see once it is running."
    )
    add("")

    # -- at a glance ------------------------------------------------------- #
    add("## At a glance")
    add("")
    add("| Approach | `app/` lines | Core symbol | The trade |")
    add("|---|---:|---|---|")
    for approach in APPROACHES:
        module, symbol = entries[approach]
        project = REPO / approach / usecase
        add(
            f"| [`{approach}`](../../{approach}/{usecase}) | {app_size(project)} "
            f"| `{module}::{symbol}` | {take_for(usecase, approach)} |"
        )
    add("")
    add(
        "Line counts are non-blank, non-comment lines across `app/`, and include "
        "each project's settings, HTTP layer, and tools — not just the loop. They "
        "are a rough proxy for how much surface you own, not a scoreboard."
    )
    add("")

    # -- the code ---------------------------------------------------------- #
    add("## The core of each implementation")
    add("")
    for approach in APPROACHES:
        module, symbol = entries[approach]
        project = REPO / approach / usecase
        add(f"### {approach} — `{module}`")
        add("")
        add(f"> {take_for(usecase, approach)}")
        add("")
        add("```python")
        add(extract(project, module, symbol))
        add("```")
        add("")

    # -- measured ---------------------------------------------------------- #
    data_path = DATA_DIR / f"{usecase}.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        add("## Measured, from a real traced run")
        add("")
        add(
            f"Captured with `?trace=1` on {data['captured_at']} against "
            f"`{data['model']}`. Regenerate with "
            "`python scripts/trace_compare.py <usecase>`. Schema: "
            "[`docs/trace-format.md`](../trace-format.md)."
        )
        add("")
        add("| | " + " | ".join(f"`{a}`" for a in APPROACHES) + " |")
        add("|---|" + "---|" * len(APPROACHES))
        for label, key in (
            ("Model calls / turns", "steps"),
            ("Tool calls", "tool_calls"),
            ("Input tokens", "input_tokens"),
            ("Output tokens", "output_tokens"),
            ("Cost (USD)", "cost_usd"),
        ):
            cells = []
            for approach in APPROACHES:
                value = data["runs"].get(approach, {}).get(key)
                cells.append("not reported" if value is None else f"{value}")
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("")
        for note in data.get("notes", []):
            add(f"- {note}")
        add("")

    add("## Read next")
    add("")
    add(
        "- Each project's own README explains its approach-specific decisions.\n"
        "- [`docs/trace-format.md`](../trace-format.md) — how these runs are "
        "recorded, and why the format is OpenTelemetry-shaped.\n"
        "- [`TRACKING.md`](../../TRACKING.md) — the per-use-case suitability "
        "matrix and every defect found by running these live."
    )
    add("")
    return "\n".join(lines)


def render_index() -> str:
    lines = [
        "# Comparison pages",
        "",
        "<!-- GENERATED by scripts/compare_usecase.py. Do not edit by hand. -->",
        "",
        "One page per use case: the core of each of the four implementations, what "
        "each costs in code, and — where a traced run has been captured — what each "
        "measurably did.",
        "",
        "| # | Use case | Page | Measured |",
        "|---|---|---|---|",
    ]
    for usecase in sorted(MANIFEST):
        number = usecase.split("-", 1)[0]
        measured = "✅" if (DATA_DIR / f"{usecase}.json").exists() else "—"
        lines.append(
            f"| {number} | {TITLES[usecase]} | [`{usecase}`]({usecase}.md) | {measured} |"
        )
    lines += [
        "",
        "**Measured** means a real traced run of all four is recorded on the page "
        "(`?trace=1`, see [`../trace-format.md`](../trace-format.md)). The rest "
        "compare code and structure only — capturing runtime numbers costs API "
        "calls, so it is done deliberately per use case rather than on every build.",
        "",
        "Regenerate everything: `python scripts/compare_usecase.py --all`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usecase", nargs="?", help="e.g. 08-autonomous-react")
    parser.add_argument("--all", action="store_true", help="every manifested use case")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the generated page differs from what is committed",
    )
    args = parser.parse_args()

    if args.all:
        targets = sorted(MANIFEST)
    elif args.usecase:
        if args.usecase not in MANIFEST:
            raise SystemExit(
                f"no manifest entry for {args.usecase!r}; "
                f"known: {', '.join(sorted(MANIFEST))}"
            )
        targets = [args.usecase]
    else:
        parser.error("give a use case or --all")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for usecase in targets:
        rendered = render(usecase)
        out = OUT_DIR / f"{usecase}.md"
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != rendered:
                print(f"STALE  {out.relative_to(REPO)} — re-run this script")
                failed = True
            else:
                print(f"ok     {out.relative_to(REPO)}")
        else:
            out.write_text(rendered, encoding="utf-8", newline="\n")
            print(f"wrote  {out.relative_to(REPO)}")

    # The index lists every manifested use case, so it is only meaningful after a
    # full pass — regenerating a single page must not truncate it.
    if args.all:
        index = OUT_DIR / "README.md"
        rendered_index = render_index()
        if args.check:
            current = index.read_text(encoding="utf-8") if index.exists() else ""
            if current != rendered_index:
                print(f"STALE  {index.relative_to(REPO)} — re-run this script")
                failed = True
            else:
                print(f"ok     {index.relative_to(REPO)}")
        else:
            index.write_text(rendered_index, encoding="utf-8", newline="\n")
            print(f"wrote  {index.relative_to(REPO)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
