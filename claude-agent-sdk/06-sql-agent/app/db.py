# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""Bundled SQLite sample database plus the read-only guard.

The database is generated from :data:`SEED_SQL` on first use rather than
committed as a binary, so a clone is reproducible and nothing opaque lands in
git.

**Two independent defences** protect the database, because the SQL being run is
model-authored:

1. :func:`assert_read_only` — a syntactic check that rejects anything that is
   not a single ``SELECT``/``WITH`` statement.
2. :func:`connect_read_only` — SQLite is opened with the ``mode=ro`` URI flag, so
   even a write that slipped past (1) fails at the driver.

Defence (1) alone would be a blocklist, which is the weaker pattern; (2) is the
one that actually holds.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "shop.db"

SEED_SQL = """
CREATE TABLE customers (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    country      TEXT NOT NULL,
    signed_up_on TEXT NOT NULL
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    REAL NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    ordered_on  TEXT NOT NULL
);

INSERT INTO customers (id, name, country, signed_up_on) VALUES
    (1, 'Ada Lovelace',   'UK',    '2025-01-12'),
    (2, 'Grace Hopper',   'US',    '2025-02-03'),
    (3, 'Alan Turing',    'UK',    '2025-02-19'),
    (4, 'Radia Perlman',  'US',    '2025-04-27'),
    (5, 'Vint Cerf',      'US',    '2025-06-08');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Robot Vacuum',      'home',    249.00),
    (2, 'Standing Desk',     'office',  399.50),
    (3, 'Mechanical Keyboard','office',  129.99),
    (4, 'Espresso Machine',  'kitchen', 549.00),
    (5, 'Desk Lamp',         'office',   39.95);

INSERT INTO orders (id, customer_id, product_id, quantity, ordered_on) VALUES
    (1, 1, 2, 1, '2025-03-01'),
    (2, 1, 3, 2, '2025-03-01'),
    (3, 2, 1, 1, '2025-03-14'),
    (4, 3, 4, 1, '2025-04-02'),
    (5, 4, 3, 1, '2025-05-11'),
    (6, 5, 5, 3, '2025-06-20'),
    (7, 2, 2, 1, '2025-07-04'),
    (8, 1, 5, 1, '2025-07-09');
"""

# Anything that is not a read. Checked as whole words so a column named
# `updated_on` does not trip the `UPDATE` rule.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Raised when model-authored SQL is not a plain read."""


def ensure_db(path: Path | None = None) -> Path:
    """Create the sample database if it does not exist. Returns its path."""
    target = path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    conn = sqlite3.connect(target)
    try:
        conn.executescript(SEED_SQL)
        conn.commit()
    finally:
        conn.close()
    return target


def assert_read_only(sql: str) -> str:
    """Validate that ``sql`` is a single read statement, or raise.

    Returns the normalised statement (trailing semicolon stripped).
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise UnsafeSQLError("empty statement")

    # Reject stacked statements outright — the classic way a guard is bypassed.
    if ";" in stripped:
        raise UnsafeSQLError("multiple statements are not allowed")

    if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise UnsafeSQLError("only SELECT (or WITH ... SELECT) statements are allowed")

    if _FORBIDDEN.search(stripped):
        raise UnsafeSQLError("statement contains a non-read keyword")

    return stripped


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open the database read-only at the driver level."""
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(path: Path) -> list[str]:
    """The real table names — which is also the allow-list `describe_table` uses."""
    with connect_read_only(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


def describe_table(path: Path, table: str) -> list[dict[str, Any]]:
    """Column metadata for one table.

    ``table`` is model-supplied, so it is matched against the real table list
    rather than interpolated into SQL.
    """
    if table not in list_tables(path):
        raise UnsafeSQLError(f"unknown table: {table}")
    with connect_read_only(path) as conn:
        # Safe: `table` is now known to be one of our own table names.
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [{"name": r["name"], "type": r["type"], "notnull": bool(r["notnull"])} for r in rows]


def run_select(path: Path, sql: str, limit: int = 50) -> list[dict[str, Any]]:
    """Run one model-written read query, behind both defences.

    `assert_read_only` rejects anything that is not a single SELECT/WITH — a
    blocklist, and therefore the weaker half. The connection is opened `mode=ro`,
    so a write that got past the parser still fails at the driver. Rows are
    capped by `limit` so a `SELECT *` on a large table cannot exhaust memory or
    the context window.
    """
    statement = assert_read_only(sql)
    with connect_read_only(path) as conn:
        rows = conn.execute(statement).fetchmany(limit)
    return [dict(r) for r in rows]
