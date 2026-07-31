# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""Unit tests for UC06 sql-agent (claude-agent-sdk). Stubbed agent, no network.

The read-only guard gets the most attention: it stands between model-authored
SQL and the database, so it is tested with injection-shaped input, and the
driver-level `mode=ro` defence is tested separately from the syntactic one.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import db
from app.agent import AgentResult, ToolCall
from app.main import create_app
from app.settings import Settings
from app.sql_agent import ask, run_select_tool, set_db_path

ANSWER = "Ada Lovelace has placed the most orders (3)."


@pytest.fixture
def sample_db(tmp_path):
    path = db.ensure_db(tmp_path / "shop.db")
    set_db_path(path)
    return path


def make_runner(tool_calls=None):
    async def runner(prompt, options) -> AgentResult:
        return AgentResult(
            text=ANSWER,
            tool_calls=tool_calls
            if tool_calls is not None
            else [
                ToolCall(name="mcp__sql__list_tables", input={}),
                ToolCall(name="mcp__sql__describe_table", input={"table": "orders"}),
                ToolCall(
                    name="mcp__sql__run_select",
                    input={"sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY 1"},
                ),
            ],
            num_turns=5,
            cost_usd=0.003,
        )

    return runner


def make_client(**kw) -> TestClient:
    return TestClient(create_app(runner=make_runner(**kw)))


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "approach": "claude-agent-sdk",
        "usecase": "06-sql-agent",
    }


def test_run_returns_answer_and_executed_queries():
    body = make_client().post("/run", json={"question": "Who orders most?"}).json()
    assert body["answer"] == ANSWER
    assert body["queries"] == [
        "SELECT customer_id, COUNT(*) FROM orders GROUP BY 1"
    ]
    assert body["tools_used"] == ["list_tables", "describe_table", "run_select"]


def test_schema_endpoint_lists_tables_and_columns():
    body = make_client().get("/schema").json()
    assert {"customers", "products", "orders"} <= set(body)
    assert any(c["name"] == "quantity" for c in body["orders"])


# --- the read-only guard ------------------------------------------------------


def test_select_is_allowed(sample_db):
    rows = db.run_select(sample_db, "SELECT name FROM customers ORDER BY id")
    assert rows[0]["name"] == "Ada Lovelace"


def test_with_cte_is_allowed(sample_db):
    rows = db.run_select(
        sample_db,
        "WITH c AS (SELECT country, COUNT(*) n FROM customers GROUP BY country) "
        "SELECT * FROM c ORDER BY n DESC",
    )
    assert rows[0]["n"] == 3  # three US customers


def test_trailing_semicolon_is_tolerated(sample_db):
    assert db.run_select(sample_db, "SELECT 1 AS x;")[0]["x"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DELETE FROM orders",
        "UPDATE products SET price = 0",
        "INSERT INTO customers VALUES (9, 'x', 'US', '2026-01-01')",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "PRAGMA table_info('customers')",
        "SELECT 1; DROP TABLE customers",  # stacked statements
        "",
        "   ",
    ],
)
def test_non_read_statements_are_rejected(sample_db, sql):
    with pytest.raises(db.UnsafeSQLError):
        db.run_select(sample_db, sql)


def test_column_named_like_a_keyword_is_not_falsely_rejected(sample_db):
    """`updated_on`-style names must not trip the whole-word UPDATE rule."""
    assert db.assert_read_only("SELECT ordered_on FROM orders") == (
        "SELECT ordered_on FROM orders"
    )


def test_driver_level_readonly_blocks_writes_even_if_guard_bypassed(sample_db):
    """Second line of defence: the connection itself refuses writes."""
    conn = db.connect_read_only(sample_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM orders")
    finally:
        conn.close()


def test_describe_table_rejects_unknown_table(sample_db):
    with pytest.raises(db.UnsafeSQLError):
        db.describe_table(sample_db, "customers'; DROP TABLE customers--")


def test_row_limit_is_enforced(sample_db):
    rows = db.run_select(sample_db, "SELECT * FROM orders", limit=3)
    assert len(rows) == 3


@pytest.mark.anyio
async def test_run_select_tool_reports_rejection_as_error_result(sample_db):
    """A rejected query must be recoverable, not fatal."""
    out = await run_select_tool.handler({"sql": "DROP TABLE customers"})
    assert out["is_error"] is True
    assert "Rejected" in out["content"][0]["text"]


@pytest.mark.anyio
async def test_run_select_tool_reports_bad_sql_as_error_result(sample_db):
    out = await run_select_tool.handler({"sql": "SELECT nope FROM customers"})
    assert out["is_error"] is True
    assert "SQL error" in out["content"][0]["text"]


@pytest.mark.anyio
async def test_options_register_sql_tools_only():
    seen = {}

    async def spy(prompt, options) -> AgentResult:
        seen["tools"] = options.allowed_tools
        return AgentResult(text="ok")

    await ask("q", Settings(), spy)
    assert seen["tools"] == [
        "mcp__sql__list_tables",
        "mcp__sql__describe_table",
        "mcp__sql__run_select",
    ]
    assert not any(t in seen["tools"] for t in ("Bash", "Write"))


# --------------------------------------------------------------------------- #
# stop_reason -- a capped run must be distinguishable from a complete one
# --------------------------------------------------------------------------- #
RUN_PAYLOAD = {"question": "q"}


def _stub_runner(**fields):
    """A runner that returns exactly the AgentResult it is handed."""

    async def runner(prompt, options) -> AgentResult:
        return AgentResult(**fields)

    return runner


def test_stop_reason_reports_a_completed_run():
    """The SDK reports no stop reason on several paths, so the field would be
    null exactly when the run was fine. `end_turn` fills that gap: callers get
    one field that is always present and always means something."""
    client = TestClient(create_app(runner=_stub_runner(text="done", num_turns=1)))
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "end_turn"


def test_stop_reason_reports_a_capped_run():
    """The gap this closes. A run cut short by `max_turns` still answers 200
    with whatever it managed to produce -- previously indistinguishable from a
    run that finished properly, which is the one thing a caller must be able to
    tell apart."""
    client = TestClient(
        create_app(
            runner=_stub_runner(
                text="partial", num_turns=8, is_error=True, stop_reason="max_turns"
            )
        )
    )
    body = client.post("/run", json=RUN_PAYLOAD).json()
    assert body["stop_reason"] == "max_turns"
