# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langgraph). See langgraph/06-sql-agent/README.md
r"""SQL agent core for UC6 as a tiny langgraph StateGraph.

The graph models the agent's control flow explicitly:

    generate -> validate --(ok)--> execute -> END
                         \--(bad)--> END   (state carries the rejection)

- ``generate`` calls the LLM for ONE SQL statement.
- ``validate`` runs the load-bearing safety check (single read-only SELECT). On
  failure it sets ``error`` in the state and a conditional edge skips execution.
- ``execute`` runs the validated SELECT under a read-only sqlite authorizer.

This is intentionally small; the value of the graph here is making the
"validate before you execute" branch a first-class, inspectable edge rather than
a buried ``if``. The LLM and DB connection are injected, so unit tests run
offline with ``FakeListChatModel`` + an in-memory fixture DB.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm import strip_thinking, system_prompt
from .settings import Settings, get_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_PATH = DATA_DIR / "seed.sql"

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
    "GRANT", "REVOKE", "MERGE", "REINDEX",
)

SYSTEM_INSTRUCTION = (
    "You are a careful SQL analyst for a SQLite database. "
    "Given the schema and a question, write EXACTLY ONE read-only SQL SELECT "
    "statement that answers it. Rules: use only SELECT; never modify data; "
    "do not write multiple statements; do not add explanations. "
    "Return ONLY the SQL, optionally inside a ```sql fenced block."
)


class SQLValidationError(ValueError):
    """Raised when a generated statement is not a single read-only SELECT."""


# --------------------------------------------------------------------------- #
# Database build + schema
# --------------------------------------------------------------------------- #
def build_db(seed_path: Path = SEED_PATH) -> sqlite3.Connection:
    """Create an in-memory SQLite DB from the seed script and return the conn."""
    # check_same_thread=False: FastAPI dispatches sync endpoints to a threadpool.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(seed_path.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def schema_text(conn: sqlite3.Connection) -> str:
    """Render a compact CREATE-TABLE-style schema string for the prompt."""
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return "\n\n".join((r["sql"] or "").strip() for r in rows)


# --------------------------------------------------------------------------- #
# SQL extraction + safety validation (CRITICAL)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply, stripping any ```sql fence."""
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    return candidate.strip()


def _strip_comments(sql: str) -> str:
    """Remove -- line comments and /* */ block comments."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def validate_select(sql: str) -> str:
    """Validate that ``sql`` is exactly one read-only SELECT.

    Returns the cleaned single statement (without a trailing ``;``) on success;
    raises :class:`SQLValidationError` otherwise. Strict by design: rejects
    writes, DDL, PRAGMA, and multiple statements even if hidden behind comments.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("empty SQL")

    cleaned = _strip_comments(sql).strip().rstrip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if ";" in cleaned:
        raise SQLValidationError("multiple statements are not allowed")

    if not cleaned:
        raise SQLValidationError("empty SQL after stripping comments")

    if not re.match(r"(?is)^\s*SELECT\b", cleaned):
        raise SQLValidationError("only single SELECT statements are allowed")

    upper = cleaned.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            raise SQLValidationError(f"forbidden keyword '{kw}' in SQL")

    return cleaned


# --------------------------------------------------------------------------- #
# Execution (read-only)
# --------------------------------------------------------------------------- #
def run_select(conn: sqlite3.Connection, sql: str, limit: int = 100) -> list[dict]:
    """Execute a validated SELECT under a read-only authorizer and return rows."""
    def _authorizer(action, *_args):
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    conn.set_authorizer(_authorizer)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(limit)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.set_authorizer(None)


def summarise(question: str, sql: str, rows: list[dict]) -> str:
    """Plain, deterministic NL summary of the result (no extra LLM call)."""
    n = len(rows)
    if n == 0:
        return f"The query for '{question}' returned no rows."
    if n == 1 and len(rows[0]) == 1:
        (value,) = rows[0].values()
        return f"The answer to '{question}' is {value}."
    return f"The query returned {n} row(s) for '{question}'."


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class SQLState(TypedDict):
    question: str
    schema: str
    sql: str
    rows: list
    explanation: str
    error: Optional[str]


def build_sql_graph(conn: sqlite3.Connection, llm: BaseChatModel,
                    settings: Settings | None = None):
    """Compile a generate -> validate -> (execute | END) graph. Deps injected."""
    settings = settings or get_settings()

    def generate(state: SQLState) -> dict:
        schema = schema_text(conn)
        messages = [
            SystemMessage(content=system_prompt(SYSTEM_INSTRUCTION, settings)),
            HumanMessage(
                content=(
                    f"Database schema:\n{schema}\n\n"
                    f"Question: {state['question']}\n\n"
                    "Write one read-only SELECT statement that answers the question."
                )
            ),
        ]
        reply = llm.invoke(messages)
        return {"schema": schema, "sql": extract_sql(strip_thinking(reply.content))}

    def validate(state: SQLState) -> dict:
        try:
            return {"sql": validate_select(state["sql"]), "error": None}
        except SQLValidationError as exc:
            return {"error": str(exc)}

    def execute(state: SQLState) -> dict:
        rows = run_select(conn, state["sql"])
        return {
            "rows": rows,
            "explanation": summarise(state["question"], state["sql"], rows),
        }

    def route_after_validate(state: SQLState) -> str:
        return "execute" if not state.get("error") else "reject"

    graph = StateGraph(SQLState)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("execute", execute)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"execute": "execute", "reject": END}
    )
    graph.add_edge("execute", END)
    return graph.compile()


def initial_state(question: str) -> SQLState:
    """A fresh graph input for a question."""
    return {
        "question": question,
        "schema": "",
        "sql": "",
        "rows": [],
        "explanation": "",
        "error": None,
    }
