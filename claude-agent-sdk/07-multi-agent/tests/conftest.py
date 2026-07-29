# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC07 Multi-agent orchestration (claude-agent-sdk). See claude-agent-sdk/07-multi-agent/README.md
"""Pytest config for the async tests.

The Agent SDK is async-first, so the parsing tests are coroutines. anyio's pytest
plugin ships with anyio (already a Starlette dependency), which avoids adding
pytest-asyncio just for this. Pinning the backend to asyncio stops anyio from
also parametrising every async test over trio, which is not installed.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
