# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
"""Audit comment coverage across every example project.

This repo is meant to be *read*, so a missing docstring is a defect in the
product, not a style nit. The check is deliberately coarse: it reports what is
undocumented and what is thin, and leaves judgement to a person.

Two things it does NOT claim to measure:

* **accuracy** — the worst comment found in this repo was not missing, it was
  wrong ("Scopes the agent to the corpus: it can only search what lives here",
  above code a live agent walked straight out of). No counter catches that;
  only running the thing does.
* **usefulness** — density is a floor, not a target. A file restating its own
  code on every line would score well here and read badly.

    python scripts/comment_audit.py            # summary + anything flagged
    python scripts/comment_audit.py --check    # exit 1 if a public symbol is bare
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
APPROACHES = ("raw-api", "langchain", "langgraph", "claude-agent-sdk")

EXEMPT_PREFIXES = ("_", "test_")

# Naming these "undocumented" produced a 54% score and would have driven someone
# to write `"""The run request."""` on `class RunRequest` 44 times. They are
# exempt because their meaning is already carried elsewhere, not because
# documentation is optional:
#
#   * route handlers  — the `@app.post("/run")` above them is the description;
#   * `create_app` / `lifespan` / `get_settings` — FastAPI conventions a reader
#     already knows, and each module's docstring covers the wiring;
#   * request/response models — the field names and types are the contract, and
#     anything non-obvious about them belongs in a field comment, which the
#     "documented" test below accepts as evidence.
EXEMPT_NAMES = frozenset({"create_app", "lifespan", "get_settings"})
SELF_DESCRIBING_BASES = frozenset({"BaseModel", "BaseSettings"})


class FileReport:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.code_lines = 0
        self.comment_lines = 0
        self.module_docstring = False
        self.bare_public: list[str] = []
        self.documented = 0
        self.total_public = 0

    @property
    def density(self) -> float:
        """Comment lines as a share of code lines (docstrings excluded)."""
        return self.comment_lines / self.code_lines if self.code_lines else 0.0

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO)).replace("\\", "/")


def _decorated_route(node: ast.AST) -> bool:
    """True for `@app.get(...)` / `@app.post(...)` handlers."""
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr in {
            "get", "post", "put", "delete", "patch", "websocket"
        }:
            return True
    return False


def _tool_described(node: ast.AST) -> bool:
    """True for `@tool("name", "what it does", schema)` MCP handlers.

    That description is not a substitute for a docstring — it is a better one.
    It is the text the *model* reads when deciding whether to call the tool, so
    it has to be accurate or the agent misbehaves, and it sits where a reader
    looking at the tool will see it.
    """
    for dec in getattr(node, "decorator_list", []):
        if (
            isinstance(dec, ast.Call)
            and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "tool"
            and len(dec.args) >= 2
            and isinstance(dec.args[1], ast.Constant)
            and isinstance(dec.args[1].value, str)
            and dec.args[1].value.strip()
        ):
            return True
    return False


def _self_describing_model(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        isinstance(b, ast.Name) and b.id in SELF_DESCRIBING_BASES for b in node.bases
    )


def _needs_doc(node: ast.AST, nested: set[int]) -> bool:
    name = getattr(node, "name", "")
    return not (
        name.startswith(EXEMPT_PREFIXES)
        or name in EXEMPT_NAMES
        or id(node) in nested          # closures live inside a documented parent
        or _decorated_route(node)
        or _tool_described(node)
        or _self_describing_model(node)
    )


def _has_explaining_comment(lines: list[str], node: ast.AST) -> bool:
    """A comment directly above the definition, or inside a class body.

    Counted as documentation because it *is* documentation. `Settings` carries
    per-field comments explaining each env var, which serves a reader better
    than a class docstring restating the class name would.
    """
    start = node.lineno - 1
    for dec in getattr(node, "decorator_list", []):
        start = min(start, dec.lineno - 1)
    i = start - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0 and lines[i].strip().startswith("#"):
        return True
    if isinstance(node, ast.ClassDef):
        body = lines[node.lineno: getattr(node, "end_lineno", node.lineno)]
        return any(ln.strip().startswith("#") for ln in body)
    return False


def audit_file(path: pathlib.Path) -> FileReport:
    r = FileReport(path)
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            r.comment_lines += 1
        else:
            r.code_lines += 1

    tree = ast.parse(src)
    r.module_docstring = ast.get_docstring(tree) is not None

    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    nested = {
        id(child)
        for node in ast.walk(tree) if isinstance(node, defs)
        for child in ast.walk(node) if isinstance(child, defs) and child is not node
    }

    for node in ast.walk(tree):
        if not isinstance(node, defs) or not _needs_doc(node, nested):
            continue
        r.total_public += 1
        if ast.get_docstring(node) or _has_explaining_comment(lines, node):
            r.documented += 1
        else:
            r.bare_public.append(f"{node.name}:{node.lineno}")
    return r


def collect() -> list[FileReport]:
    reports = []
    for approach in APPROACHES:
        for path in sorted((REPO / approach).rglob("app/*.py")):
            if path.name == "__init__.py" or ".venv" in path.parts:
                continue
            reports.append(audit_file(path))
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any public symbol or module lacks a docstring")
    ap.add_argument("--thin", type=float, default=0.08,
                    help="flag files whose comment density is below this")
    args = ap.parse_args()

    reports = collect()
    code = sum(r.code_lines for r in reports)
    comments = sum(r.comment_lines for r in reports)
    public = sum(r.total_public for r in reports)
    documented = sum(r.documented for r in reports)
    no_module_doc = [r for r in reports if not r.module_docstring]
    bare = [r for r in reports if r.bare_public]
    thin = [r for r in reports if r.density < args.thin]

    print(f"files            : {len(reports)}")
    print(f"code lines       : {code}")
    print(f"comment lines    : {comments}  ({comments / code:.1%} of code)")
    print(f"public symbols   : {public}")
    print(f"  with docstring : {documented}  ({documented / public:.1%})")
    print(f"module docstrings: {len(reports) - len(no_module_doc)}/{len(reports)}")

    for label, group in (("missing module docstring", no_module_doc),
                         ("undocumented public symbols", bare)):
        if group:
            print(f"\n{label}:")
            for r in group:
                detail = ", ".join(r.bare_public) if r.bare_public else ""
                print(f"  {r.rel} {detail}")

    if thin:
        print(f"\nthin (< {args.thin:.0%} inline comments) — review, not necessarily wrong:")
        for r in sorted(thin, key=lambda x: x.density):
            print(f"  {r.density:5.1%}  {r.rel}")

    if args.check and (no_module_doc or bare):
        print("\nFAIL: something public is undocumented.")
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
