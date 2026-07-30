# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — langgraph project template. See langgraph/_template/README.md
"""Minimal langgraph FastAPI skeleton.

A tiny single-node ``StateGraph`` that echoes the question through the LLM.
This is the Phase 0 starting point other use cases copy and extend.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TypedDict

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from .llm import build_llm, strip_thinking, system_prompt

APPROACH = "langgraph"
USECASE = "_template"


class RunRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    snippet: str


class RunResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class GraphState(TypedDict):
    question: str
    answer: str


def build_graph(llm: BaseChatModel):
    """Build a one-node graph: ``echo`` -> END. ``llm`` is injectable for tests."""

    def echo(state: GraphState) -> GraphState:
        messages = [
            SystemMessage(content=system_prompt("You are a concise echo assistant.")),
            HumanMessage(content=state["question"]),
        ]
        result = llm.invoke(messages)
        return {
            "question": state["question"],
            "answer": strip_thinking(result.content),
        }

    graph = StateGraph(GraphState)
    graph.add_node("echo", echo)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    return graph.compile()


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "graph"):
            app.state.graph = build_graph(build_llm())
        yield

    app = FastAPI(title="langgraph _template", lifespan=lifespan)
    # Build eagerly when an LLM is injected (tests) or fall back at startup.
    if llm is not None:
        app.state.graph = build_graph(llm)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        out = app.state.graph.invoke({"question": req.question, "answer": ""})
        return RunResponse(answer=out["answer"], sources=[])

    return app


app = create_app()
