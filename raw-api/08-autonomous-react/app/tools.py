# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC8 Autonomous ReAct (raw-api). See raw-api/08-autonomous-react/README.md
"""Deterministic, offline tools for the ReAct agent.

Two tools, both pure Python with NO network and NO ``eval``:

- ``calculator(expression)`` — safe arithmetic evaluated by walking a parsed
  ``ast`` tree. Only numbers and the basic arithmetic operators are allowed, so
  hostile input like ``__import__('os').system('...')`` is rejected outright.
- ``search(query)`` — keyword lookup over the bundled ``data/facts.md`` (a few
  ``key: value`` facts). Returns the best-matching line(s).

Each tool is a plain callable taking a single string and returning a string
(the Observation the loop threads back to the model).
"""
from __future__ import annotations

import ast
import operator
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FACTS_FILE = DATA_DIR / "facts.md"


# --------------------------------------------------------------------------- #
# calculator — safe ast-based arithmetic (NO eval)
# --------------------------------------------------------------------------- #
class UnsafeExpression(ValueError):
    """Raised when an expression contains anything but plain arithmetic."""


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    # Python 3.8+: numeric literals are ast.Constant.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpression(f"non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    # Anything else (Name, Call, Attribute, Subscript, ...) is rejected.
    raise UnsafeExpression(f"disallowed syntax: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression safely. Raises UnsafeExpression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"could not parse: {exc}") from exc
    return _eval_node(tree)


def calculator(expression: str) -> str:
    """Tool entry point: evaluate arithmetic and return a clean string result."""
    value = safe_eval(expression.strip())
    # Render whole floats without a trailing ".0" so "30*2" -> "60".
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --------------------------------------------------------------------------- #
# search — keyword lookup over data/facts.md
# --------------------------------------------------------------------------- #
def load_facts(path: Path = FACTS_FILE) -> list[str]:
    """Return the non-heading, non-blank fact lines from the corpus."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _score(query: str, line: str) -> int:
    """Word-overlap score between query and a fact line (case-insensitive)."""
    q_words = {w for w in query.lower().split() if w}
    l_words = {w.strip(":,.") for w in line.lower().split()}
    return len(q_words & l_words)


def search(query: str, path: Path = FACTS_FILE) -> str:
    """Tool entry point: return the best-matching fact line(s) for a query."""
    facts = load_facts(path)
    if not facts:
        return "No facts available."
    scored = sorted(facts, key=lambda ln: _score(query, ln), reverse=True)
    best = _score(query, scored[0])
    if best == 0:
        return "No matching fact found."
    # Return all lines tying for the top score (usually just one).
    top = [ln for ln in scored if _score(query, ln) == best]
    return " | ".join(top[:3])


# Registry consumed by the ReAct loop: name -> (callable, description).
TOOLS: dict[str, tuple] = {
    "calculator": (
        calculator,
        "Evaluate a single arithmetic expression, e.g. '30 * 2'. "
        "Input: the expression. Only numbers and + - * / // % ** are allowed.",
    ),
    "search": (
        search,
        "Look up a fact from the Northwind knowledge base by keywords, "
        "e.g. 'return window'. Input: a short query.",
    ),
}
