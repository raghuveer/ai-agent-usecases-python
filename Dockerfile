# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
#
# Builds ONE example project. Which one is a build arg, because every project in
# this repo is self-contained by design — there is no shared package to install.
#
#   docker build --build-arg PROJECT=langgraph/10-hitl-approval -t uc .
#
# Only works for the three OpenAI-surface approaches. `claude-agent-sdk` needs
# Node plus the Claude Code CLI in the image and an Anthropic-compatible
# endpoint, so it is deliberately not part of the zero-install path.
FROM python:3.12-slim

ARG PROJECT=raw-api/01-rag
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

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
