# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC06 SQL / DB agent (claude-agent-sdk). See claude-agent-sdk/06-sql-agent/README.md
"""Agent wiring for the claude-agent-sdk approach.

This module holds the three pieces every use case in this approach reuses:

1. :func:`sdk_env` — translate the repo's ``LLM_*`` settings into the
   environment variables the Agent SDK subprocess reads. The SDK does not take a
   base URL or key as a Python argument; it spawns the Claude Code CLI and that
   process reads ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` from its env.
2. :func:`collect` — fold the SDK's async message stream into a plain
   :class:`AgentResult`. This is the parsing seam, and it is pure over public SDK
   types, so unit tests exercise it with real ``AssistantMessage`` /
   ``ResultMessage`` objects and never spawn the CLI.
3. :func:`default_runner` — the production runner: ``query()`` piped into
   :func:`collect`.

**The injection seam.** Every ``create_app`` takes a ``runner`` callable
defaulting to :func:`default_runner`. Unit tests pass a stub, so they are fully
offline — no CLI, no network, no key. This mirrors the injectable LLM client the
other three approaches use.

**Auth gotcha (verified against the local gateway):** the platform gateway
requires ``Authorization: Bearer <virtual-key>`` and rejects ``x-api-key``.
Setting ``ANTHROPIC_API_KEY`` makes the CLI send ``x-api-key`` (→ 401); setting
``ANTHROPIC_AUTH_TOKEN`` makes it send ``Authorization: Bearer`` (→ works). Use
``ANTHROPIC_AUTH_TOKEN``.
"""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)

from .settings import Settings


@dataclass
class ToolCall:
    """One tool invocation the agent made, flattened for the API response."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """What a single agent run produced, independent of SDK message plumbing."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    num_turns: int = 0
    cost_usd: float = 0.0
    is_error: bool = False
    stop_reason: str | None = None

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls]


# One throwaway config dir per process. Keeping the CLI's config out of the
# developer's ~/.claude is the only reliable way to stop project memory and
# CLAUDE.md leaking into agent context (see sdk_env).
_ISOLATED_CONFIG_DIR = tempfile.mkdtemp(prefix="agentsdk-config-")


def sdk_env(settings: Settings) -> dict[str, str]:
    """Environment for the Agent SDK subprocess.

    ``ANTHROPIC_AUTH_TOKEN`` (not ``ANTHROPIC_API_KEY``) — see the module
    docstring for why. Pointing ``ANTHROPIC_BASE_URL`` at the platform gateway
    keeps guardrails and usage tracking in the path; unset it in ``.env`` to talk
    to ``api.anthropic.com`` directly instead.
    """
    return {
        "ANTHROPIC_BASE_URL": settings.llm_base_url,
        "ANTHROPIC_AUTH_TOKEN": settings.llm_gateway_key,
        # The CLI otherwise emits telemetry and update checks we neither want nor
        # can reach from an air-gapped host.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # THIS is what actually isolates the run. `setting_sources=[]` gates
        # settings.json files ONLY — it does NOT stop the CLI loading the
        # developer's ~/.claude project memory or a parent CLAUDE.md. Verified
        # empirically: with setting_sources=[] set, a probe agent still recited
        # this repo's private memory index verbatim. Pointing CLAUDE_CONFIG_DIR
        # at an empty directory returns "NONE VISIBLE".
        #
        # Two reasons this matters: reproducibility (a run must not depend on
        # whose laptop it is on), and disclosure (developer memory could
        # otherwise be echoed into an API response).
        "CLAUDE_CONFIG_DIR": _ISOLATED_CONFIG_DIR,
    }


def build_options(
    settings: Settings,
    *,
    system_prompt: str,
    allowed_tools: list[str] | None = None,
    **overrides: Any,
) -> ClaudeAgentOptions:
    """Assemble ``ClaudeAgentOptions`` with this repo's budget discipline applied.

    ``setting_sources=[]`` is deliberate and load-bearing: it stops the SDK from
    reading the *developer's* ``~/.claude`` and the repo's ``.claude/`` settings,
    so a run behaves identically on a maintainer's laptop and in CI. Without it
    these examples would silently inherit local config.

    ``tools=[]`` starts from no built-in tools; each use case opts in explicitly
    so the README's tool list is the truth.
    """
    opts: dict[str, Any] = {
        "system_prompt": system_prompt,
        "model": settings.llm_model,
        "env": sdk_env(settings),
        "max_turns": settings.agent_max_turns,
        "max_budget_usd": settings.agent_max_budget_usd,
        "effort": settings.agent_effort,
        "allowed_tools": allowed_tools or [],
        "setting_sources": [],
        "permission_mode": "bypassPermissions",
    }
    opts.update(overrides)
    return ClaudeAgentOptions(**opts)


async def collect(messages: AsyncIterable[Any]) -> AgentResult:
    """Fold the SDK's message stream into an :class:`AgentResult`.

    Assistant text accumulates across turns; ``ResultMessage`` carries the run
    totals. ``ThinkingBlock`` content is deliberately dropped — it is reasoning,
    not answer, and echoing it into an API response would leak chain-of-thought
    to callers.
    """
    result = AgentResult()
    chunks: list[str] = []

    async for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    result.tool_calls.append(
                        ToolCall(name=block.name, input=dict(block.input or {}))
                    )
                elif isinstance(block, ThinkingBlock):
                    continue
        elif isinstance(message, ResultMessage):
            result.num_turns = message.num_turns
            result.cost_usd = message.total_cost_usd or 0.0
            result.is_error = bool(message.is_error)
            result.stop_reason = message.stop_reason
            # `result` is the CLI's own final text; prefer it when the assistant
            # emitted none (e.g. the run ended on a tool result).
            if not chunks and message.result:
                chunks.append(message.result)

    result.text = "\n".join(c for c in chunks if c).strip()
    return result


# The SDK raises on terminal conditions rather than yielding a ResultMessage, so
# the caps we deliberately configure arrive as exception text. Map them back.
_CAP_MARKERS = (
    ("Reached maximum number of turns", "max_turns"),
    ("Reached maximum budget", "max_budget"),
)


def _cap_reason(message: str) -> str | None:
    for marker, reason in _CAP_MARKERS:
        if marker in message:
            return reason
    return None


async def stream_prompt(prompt: str) -> AsyncIterator[dict[str, Any]]:
    """Wrap a prompt as the SDK's streaming-input user message.

    Required whenever ``can_use_tool`` is set: the permission callback is only
    supported in streaming mode, and passing a bare string raises
    ``ValueError: can_use_tool callback requires streaming mode``. The dict
    shape mirrors what the SDK itself writes for string prompts.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
    }


async def default_runner(prompt: str, options: ClaudeAgentOptions) -> AgentResult:
    """Production runner: drive a real agent loop via the SDK.

    Selects streaming input when a permission callback is configured, because
    the SDK rejects a string prompt in that mode.
    """
    stream = (
        query(prompt=stream_prompt(prompt), options=options)
        if options.can_use_tool is not None
        else query(prompt=prompt, options=options)
    )
    try:
        return await collect(stream)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a known cap
        reason = _cap_reason(str(exc))
        if reason is None:
            raise
        # Hitting a cap we configured is an expected terminal condition, not a
        # failure: report it so callers can surface "incomplete" instead of 500.
        return AgentResult(is_error=True, stop_reason=reason)


# The seam unit tests replace.
Runner = Callable[[str, ClaudeAgentOptions], Awaitable[AgentResult]]
