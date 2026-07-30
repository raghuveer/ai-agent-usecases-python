# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC6 SQL / DB agent (langgraph). See langgraph/06-sql-agent/README.md
"""Unit tests for UC6 sql-agent (langgraph). Fake LLM + fixture DB, no network.

The SQL safety validator is the critical piece, so it gets the most coverage. The
compiled graph is exercised too: a SELECT routes generate -> validate -> execute,
and a DELETE routes generate -> validate -> reject (error set, no execution).
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import sqlagent
from app.llm import system_prompt
from app.main import create_app
from app.settings import Settings
from app.sqlagent import build_db, build_sql_graph, initial_state


def fixture_db() -> sqlite3.Connection:
    return build_db()


def make_client(reply: str) -> TestClient:
    conn = fixture_db()
    llm = FakeListChatModel(responses=[reply])
    return TestClient(create_app(conn=conn, llm=llm))


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def test_validate_allows_plain_select():
    assert sqlagent.validate_select("SELECT * FROM customers") == "SELECT * FROM customers"


def test_validate_strips_trailing_semicolon():
    assert sqlagent.validate_select("SELECT COUNT(*) FROM orders;") == "SELECT COUNT(*) FROM orders"


def test_validate_allows_join_and_aggregate():
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
        "SELECT 1; DROP TABLE customers",
        "SELECT 1; SELECT 2",
        "SELECT * FROM customers; DELETE FROM orders",
        "",
        "   ",
        "-- SELECT 1\nDELETE FROM customers",
        "SELECT 1 /* ; DROP TABLE customers */ ; DROP TABLE orders",
    ],
)
def test_validate_rejects_non_select(bad):
    with pytest.raises(sqlagent.SQLValidationError):
        sqlagent.validate_select(bad)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_sql_from_fence():
    reply = "Sure:\n```sql\nSELECT COUNT(*) FROM customers\n```"
    assert sqlagent.extract_sql(reply) == "SELECT COUNT(*) FROM customers"


def test_extract_sql_without_fence():
    assert sqlagent.extract_sql("SELECT 1") == "SELECT 1"


# --------------------------------------------------------------------------- #
# DB + execution
# --------------------------------------------------------------------------- #
def test_build_db_has_seed_rows():
    conn = fixture_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 5
    finally:
        conn.close()


def test_run_select_returns_rows():
    conn = fixture_db()
    try:
        assert sqlagent.run_select(conn, "SELECT COUNT(*) AS n FROM customers") == [{"n": 5}]
    finally:
        conn.close()


def test_run_select_authorizer_blocks_writes():
    conn = fixture_db()
    try:
        with pytest.raises(sqlite3.DatabaseError):
            sqlagent.run_select(conn, "DELETE FROM customers")
    finally:
        conn.close()


def test_no_think_prefix_for_qwen3():
    s_qwen = Settings(llm_model="qwen3:1.7b")
    s_coder = Settings(llm_model="qwen-local-coder")
    assert system_prompt("x", s_qwen).startswith("/no_think")
    assert system_prompt("x", s_coder) == "x"


# --------------------------------------------------------------------------- #
# Graph: SELECT -> execute, DELETE -> reject
# --------------------------------------------------------------------------- #
def test_graph_select_routes_to_execute():
    conn = fixture_db()
    llm = FakeListChatModel(responses=["```sql\nSELECT COUNT(*) AS count FROM customers\n```"])
    try:
        graph = build_sql_graph(conn, llm)
        out = graph.invoke(initial_state("how many customers?"))
        assert out["error"] is None
        assert out["sql"] == "SELECT COUNT(*) AS count FROM customers"
        assert out["rows"] == [{"count": 5}]
        assert "5" in out["explanation"]
    finally:
        conn.close()


def test_graph_delete_routes_to_reject():
    conn = fixture_db()
    llm = FakeListChatModel(responses=["DELETE FROM customers"])
    try:
        graph = build_sql_graph(conn, llm)
        out = graph.invoke(initial_state("wipe it"))
        assert out["error"]              # validation rejected it
        assert out["rows"] == []         # execute node never ran
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# HTTP level
# --------------------------------------------------------------------------- #
def test_health():
    client = make_client("```sql\nSELECT 1\n```")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "langgraph",
        "usecase": "06-sql-agent",
    }


def test_run_returns_sql_rows_explanation():
    client = make_client("```sql\nSELECT COUNT(*) AS count FROM customers\n```")
    resp = client.post("/run", json={"question": "how many customers?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"] == "SELECT COUNT(*) AS count FROM customers"
    assert body["rows"] == [{"count": 5}]
    assert body["explanation"]


def test_run_returns_400_on_write():
    client = make_client("DELETE FROM customers")
    resp = client.post("/run", json={"question": "delete all"})
    assert resp.status_code == 400


def test_strip_thinking_removes_qwen3_think_blocks():
    """`/no_think` still emits an empty <think></think> pair; it must not ship.

    Found by running the Docker quickstart, whose default model is a qwen3 tag:
    answers came back with a leading empty thinking block before the text.
    """
    from app.llm import strip_thinking

    empty_block = "<think>" + "\n\n" + "</think>" + "\n\n" + "30 days."
    assert strip_thinking(empty_block) == "30 days."
    assert strip_thinking("<think>reasoning</think>Answer.") == "Answer."
    assert strip_thinking("  30 days.  ") == "30 days."
    # Chain-of-thought must never survive into a response.
    assert "reasoning" not in strip_thinking("<think>reasoning</think>ok")
