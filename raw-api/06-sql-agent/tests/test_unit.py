"""Unit tests for UC6 sql-agent (raw-api) — fully mocked, no network.

The SQL safety validator is the critical piece, so it gets the most coverage:
it must accept exactly one read-only SELECT and reject writes/DDL/multi-statement
attempts. The full pipeline is exercised against an in-memory fixture DB with a
fake LLM, asserting that a SELECT executes and a DELETE is rejected.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import llm, sqlagent
from app.main import create_app
from app.settings import Settings


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def fixture_db() -> sqlite3.Connection:
    """Build the bundled seed DB in memory."""
    return sqlagent.build_db()


def _fake_openai_client(reply: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def fixed_llm(reply: str):
    def _call(system_prompt: str, user_prompt: str) -> str:
        return reply
    return _call


# --------------------------------------------------------------------------- #
# Validator: accepts read-only SELECT
# --------------------------------------------------------------------------- #
def test_validate_allows_plain_select():
    assert sqlagent.validate_select("SELECT * FROM customers") == "SELECT * FROM customers"


def test_validate_strips_trailing_semicolon():
    assert sqlagent.validate_select("SELECT COUNT(*) FROM orders;") == "SELECT COUNT(*) FROM orders"


def test_validate_allows_select_with_join_and_aggregate():
    sql = "SELECT c.city, SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.city"
    assert sqlagent.validate_select(sql) == sql


@pytest.mark.parametrize(
    "bad",
    [
        "DELETE FROM customers",
        "INSERT INTO customers (id, name, city) VALUES (9, 'x', 'y')",
        "UPDATE customers SET city = 'x' WHERE id = 1",
        "DROP TABLE customers",
        "ALTER TABLE orders ADD COLUMN foo TEXT",
        "CREATE TABLE evil (id INT)",
        "PRAGMA table_info(customers)",
        "SELECT 1; DROP TABLE customers",            # multi-statement
        "SELECT 1; SELECT 2",                          # multi-statement
        "SELECT * FROM customers; DELETE FROM orders", # smuggled write
        "",                                            # empty
        "   ",                                         # blank
        "-- SELECT 1\nDELETE FROM customers",          # comment-hidden write
        "SELECT 1 /* ; DROP TABLE customers */ ; DROP TABLE orders",
    ],
)
def test_validate_rejects_non_select(bad):
    with pytest.raises(sqlagent.SQLValidationError):
        sqlagent.validate_select(bad)


def test_validate_strips_comments_but_keeps_select():
    sql = "SELECT name FROM customers -- a comment\nWHERE id = 1"
    cleaned = sqlagent.validate_select(sql)
    assert cleaned.upper().startswith("SELECT")
    assert "--" not in cleaned


# --------------------------------------------------------------------------- #
# SQL extraction from a fenced reply
# --------------------------------------------------------------------------- #
def test_extract_sql_from_fence():
    reply = "Here you go:\n```sql\nSELECT COUNT(*) FROM customers\n```\nthanks"
    assert sqlagent.extract_sql(reply) == "SELECT COUNT(*) FROM customers"


def test_extract_sql_without_fence():
    assert sqlagent.extract_sql("SELECT 1") == "SELECT 1"


# --------------------------------------------------------------------------- #
# DB build + schema + execution
# --------------------------------------------------------------------------- #
def test_build_db_has_seed_rows():
    conn = fixture_db()
    try:
        n_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert n_customers == 5
        assert n_orders == 5
    finally:
        conn.close()


def test_schema_text_lists_tables():
    conn = fixture_db()
    try:
        schema = sqlagent.schema_text(conn)
        assert "customers" in schema
        assert "orders" in schema
    finally:
        conn.close()


def test_run_select_returns_rows():
    conn = fixture_db()
    try:
        rows = sqlagent.run_select(conn, "SELECT COUNT(*) AS n FROM customers")
        assert rows == [{"n": 5}]
    finally:
        conn.close()


def test_run_select_authorizer_blocks_writes():
    """Even if validation were bypassed, the authorizer denies writes."""
    conn = fixture_db()
    try:
        with pytest.raises(sqlite3.DatabaseError):
            sqlagent.run_select(conn, "DELETE FROM customers")
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Pipeline: SELECT executes, DELETE rejected — no network
# --------------------------------------------------------------------------- #
def test_answer_runs_select_against_fixture_db():
    conn = fixture_db()
    try:
        result = sqlagent.answer(
            "How many customers are there?",
            conn=conn,
            llm_call=fixed_llm("```sql\nSELECT COUNT(*) AS count FROM customers\n```"),
        )
        assert result["sql"] == "SELECT COUNT(*) AS count FROM customers"
        assert result["rows"] == [{"count": 5}]
        assert "5" in result["explanation"]
    finally:
        conn.close()


def test_answer_rejects_delete():
    conn = fixture_db()
    try:
        with pytest.raises(sqlagent.SQLValidationError):
            sqlagent.answer(
                "wipe everything",
                conn=conn,
                llm_call=fixed_llm("DELETE FROM customers"),
            )
    finally:
        conn.close()


def test_apply_no_think_qwen3_only():
    assert llm.apply_no_think("qwen3:1.7b", "x").startswith("/no_think")
    assert llm.apply_no_think("qwen-local-coder", "x") == "x"


# --------------------------------------------------------------------------- #
# HTTP level (mocked DB + mocked openai client, no network)
# --------------------------------------------------------------------------- #
def test_health_and_run_offline():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client(
        "```sql\nSELECT COUNT(*) AS count FROM customers\n```"
    )
    # Pre-set the DB so the startup hook does NOT rebuild it (and tests stay fast).
    app.state.conn = fixture_db()

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json() == {"status": "ok", "approach": "raw-api", "usecase": "06-sql-agent"}

    r = client.post("/run", json={"question": "how many customers?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sql"] == "SELECT COUNT(*) AS count FROM customers"
    assert body["rows"] == [{"count": 5}]


def test_run_returns_400_on_write():
    app = create_app()
    app.state.settings = Settings()
    app.state.client = _fake_openai_client("DELETE FROM customers")
    app.state.conn = fixture_db()

    client = TestClient(app)
    r = client.post("/run", json={"question": "delete all customers"})
    assert r.status_code == 400
