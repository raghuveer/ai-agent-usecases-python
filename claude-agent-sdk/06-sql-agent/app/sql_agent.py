# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""Natural-language questions answered against SQLite, via custom SDK tools.

The agent gets three tools and works out the rest: look at the schema, write a
query, run it, and read the answer off the rows. Schema discovery is *its* job,
not a prompt-stuffing exercise — which is the difference from the other
approaches, where the schema is typically injected wholesale into the system
prompt whether or not the question needs it.

    list_tables ──► describe_table ──► run_select ──► answer

Every query goes through the read-only guard in :mod:`app.db`. A rejected query
comes back as an error tool result, so the agent can correct itself rather than
the request failing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import db
from .agent import Runner, build_options, default_runner
from .settings import Settings

# Resolved once at import; tools close over it.
_DB_PATH: Path = db.ensure_db()


def set_db_path(path: Path) -> None:
    """Point the tools at another database (used by tests)."""
    global _DB_PATH
    _DB_PATH = path


@tool("list_tables", "List the tables in the database.", {})
async def list_tables_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": ", ".join(db.list_tables(_DB_PATH))}]
    }


@tool("describe_table", "Show the columns and types of one table.", {"table": str})
async def describe_table_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        columns = db.describe_table(_DB_PATH, str(args.get("table", "")))
    except db.UnsafeSQLError as exc:
        return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}
    return {"content": [{"type": "text", "text": json.dumps(columns)}]}


@tool(
    "run_select",
    "Run a read-only SELECT query and return up to 50 rows as JSON.",
    {"sql": str},
)
async def run_select_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        rows = db.run_select(_DB_PATH, str(args.get("sql", "")))
    except db.UnsafeSQLError as exc:
        # Recoverable: tell the agent why, let it rewrite the query.
        return {
            "content": [{"type": "text", "text": f"Rejected: {exc}"}],
            "is_error": True,
        }
    except Exception as exc:  # malformed SQL, unknown column, ...
        return {
            "content": [{"type": "text", "text": f"SQL error: {exc}"}],
            "is_error": True,
        }
    return {"content": [{"type": "text", "text": json.dumps(rows, default=str)}]}


SQL_TOOLS = [
    "mcp__sql__list_tables",
    "mcp__sql__describe_table",
    "mcp__sql__run_select",
]

SYSTEM_PROMPT = """You answer questions about a shop database.

Method:
1. Call list_tables to see what exists.
2. Call describe_table for the tables you need — do not guess column names.
3. Write a single SELECT and run it with run_select.
4. Answer from the rows you got back.

Rules:
- Read-only: SELECT (or WITH ... SELECT) only. Writes are rejected.
- One statement per call; no semicolon-separated statements.
- If a query is rejected or errors, read the message and fix the query.
- State the answer plainly, with the numbers from the rows."""


def build_sql_server():
    return create_sdk_mcp_server(
        name="sql",
        version="1.0.0",
        tools=[list_tables_tool, describe_table_tool, run_select_tool],
    )


@dataclass
class SqlResult:
    answer: str
    queries: list[str]
    tools_used: list[str]
    num_turns: int
    cost_usd: float


async def ask(
    question: str, settings: Settings, runner: Runner | None = None
) -> SqlResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=SQL_TOOLS,
        mcp_servers={"sql": build_sql_server()},
        # Schema discovery then query then possibly a fix — needs headroom.
        max_turns=max(settings.agent_max_turns, 8),
    )
    result = await runner(question, options)

    return SqlResult(
        answer=result.text,
        queries=[
            str(c.input.get("sql", ""))
            for c in result.tool_calls
            if c.name.endswith("run_select") and c.input.get("sql")
        ],
        tools_used=[c.name.split("__")[-1] for c in result.tool_calls],
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
    )
