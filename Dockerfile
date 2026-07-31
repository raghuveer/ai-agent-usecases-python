# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
#
# Builds ONE example project. Which one is a build arg, because every project in
# this repo is self-contained by design — there is no shared package to install.
#
#   docker build --build-arg PROJECT=langgraph/10-hitl-approval -t uc .
#
# All four approaches build here, including `claude-agent-sdk` — which needs no
# Node and no `npm install -g @anthropic-ai/claude-code`, contrary to what this
# file used to claim. The SDK's wheel is platform-tagged and ships the CLI as a
# native binary (`claude_agent_sdk/_bundled/claude`, verified 2.1.220 in the
# built image), and the committed lockfiles carry the manylinux wheels. It does
# still need a real Anthropic-compatible endpoint and key, so it is not part of
# the zero-install `docker compose up` path — see docker-compose.agent-sdk.yml.
FROM python:3.12-slim

ARG PROJECT=raw-api/01-rag
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# NOT installed here, deliberately: bubblewrap + socat, which is how the CLI
# sandboxes bash on Linux (F9). The obvious move is to add them so the
# container's sandbox is real. Measured, that makes things worse:
#
#   * without them the CLI *downgrades* cleanly — it says so on stderr, the
#     response reports `sandboxed: false`, and the agent works normally
#     (4 turns, $0.16 on a slugify task);
#   * with them installed but unusable, every single Bash call dies on
#     `bwrap: pivot_root: Operation not permitted`. The agent does not stop —
#     it flails, retrying and rewording, and gave up at 17 turns and $0.99
#     having proved nothing.
#
# bwrap needs privileges this container does not have, and `--privileged` did
# not help either (`apply-seccomp: write /proc/self/uid_map: Operation not
# permitted` under Rancher Desktop / Lima). A half-working sandbox is worse
# than an absent one that says so.
#
# For a container deployment **the container is the boundary** — which is what
# docs/security-review.md asks for anyway; the in-process sandbox is defence in
# depth on a developer machine. To try it regardless, install the two packages
# here and run on a host whose kernel permits unprivileged user namespaces;
# `sandboxed` in the response will tell you whether it took.

RUN pip install --no-cache-dir uv==0.11.24

WORKDIR /app

# Dependencies first, so editing app code does not re-resolve the world.
# uv.lock is committed (v0.4.0), so `--locked` gives the exact set the tests ran
# against — the image cannot silently drift from CI.
COPY ${PROJECT}/pyproject.toml ${PROJECT}/uv.lock ./
RUN uv sync --extra dev --locked --no-install-project

COPY ${PROJECT}/ ./

# Installs the project itself now that its source is present.
RUN uv sync --extra dev --locked

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
