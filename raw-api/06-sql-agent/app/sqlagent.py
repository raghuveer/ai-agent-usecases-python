# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (raw-api). See raw-api/06-sql-agent/README.md
"""Hand-written SQL agent: schema -> prompt -> ONE SQL -> validate -> execute -> summarise.

This is the raw-api point: nothing is hidden behind a framework. We build a tiny
SQLite DB from ``data/seed.sql``, introspect its schema by hand, inline that schema
into the prompt, ask the model for ONE SQL statement, then run a strict safety
validator before touching the database.

The safety validator is the load-bearing piece: it must accept exactly one
read-only ``SELECT`` and reject everything else (writes, DDL, multiple
statements, comment-smuggled writes). Both the LLM call and the DB connection are
injectable so unit tests run with no network and an in-memory fixture DB.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_PATH = DATA_DIR / "seed.sql"

# Statements that must never reach the database from a generated query.
FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
    "GRANT", "REVOKE", "MERGE", "REINDEX",
)

SYSTEM_PROMPT = (
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
    # check_same_thread=False: FastAPI dispatches sync endpoints to a threadpool,
    # so the connection is used from a different thread than it was built in.
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
    raises :class:`SQLValidationError` otherwise. This is intentionally strict:
    we reject writes, DDL, PRAGMA, and multiple statements even if they are
    hidden behind comments.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("empty SQL")

    cleaned = _strip_comments(sql).strip()

    # Allow exactly one optional trailing semicolon; reject internal ones
    # (which would chain a second statement).
    cleaned = cleaned.rstrip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if ";" in cleaned:
        raise SQLValidationError("multiple statements are not allowed")

    if not cleaned:
        raise SQLValidationError("empty SQL after stripping comments")

    # Must START with SELECT (a leading WITH ... could hide a writable CTE in
    # other engines; SQLite CTEs are read-only but we keep it simple & strict).
    if not re.match(r"(?is)^\s*SELECT\b", cleaned):
        raise SQLValidationError("only single SELECT statements are allowed")

    # No forbidden keyword anywhere (word-boundary match, case-insensitive).
    upper = cleaned.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            raise SQLValidationError(f"forbidden keyword '{kw}' in SQL")

    return cleaned


# --------------------------------------------------------------------------- #
# Execution (read-only)
# --------------------------------------------------------------------------- #
def run_select(conn: sqlite3.Connection, sql: str, limit: int = 100) -> list[dict]:
    """Execute a validated SELECT against a read-only authorizer and return rows.

    A sqlite authorizer is installed for the duration of the query so that even
    if validation were bypassed, any write/DDL is denied at the engine level.
    """
    def _authorizer(action, *_args):
        # Permit only read operations + the bookkeeping ops a SELECT needs.
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


# --------------------------------------------------------------------------- #
# Prompt + pipeline
# --------------------------------------------------------------------------- #
def build_prompt(question: str, schema: str) -> str:
    """Build the schema-injection user prompt."""
    return (
        f"Database schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "Write one read-only SELECT statement that answers the question."
    )


# An LLM call is `(system_prompt, user_prompt) -> reply`. Injectable for tests.
LLMCall = Callable[[str, str], str]


def summarise(question: str, sql: str, rows: list[dict]) -> str:
    """Plain, deterministic NL summary of the result (no extra LLM call)."""
    n = len(rows)
    if n == 0:
        return f"The query for '{question}' returned no rows."
    if n == 1 and len(rows[0]) == 1:
        (value,) = rows[0].values()
        return f"The answer to '{question}' is {value}."
    return f"The query returned {n} row(s) for '{question}'."


def answer(
    question: str,
    *,
    conn: sqlite3.Connection,
    llm_call: LLMCall,
) -> dict:
    """Full pass: schema -> prompt -> LLM -> validate -> execute -> summarise.

    Raises :class:`SQLValidationError` if the generated SQL is not a single
    read-only SELECT (the HTTP layer maps that to 400).
    """
    schema = schema_text(conn)
    user_prompt = build_prompt(question, schema)
    raw = llm_call(SYSTEM_PROMPT, user_prompt)
    sql = validate_select(extract_sql(raw))
    rows = run_select(conn, sql)
    return {
        "sql": sql,
        "rows": rows,
        "explanation": summarise(question, sql, rows),
    }
