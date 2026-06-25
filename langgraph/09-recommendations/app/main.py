"""FastAPI app for UC9 recommendations (langgraph approach).

On startup it loads the bundled catalog + profiles and compiles the recommend
graph. ``create_app`` accepts injected data + LLM so unit tests run offline. The
ranking node is deterministic plain Python; only the explain node calls the model.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .llm import build_llm
from .recommend import Item, Profile, build_recommend_graph, load_catalog, load_profiles
from .settings import get_settings

APPROACH = "langgraph"
USECASE = "09-recommendations"


class RunRequest(BaseModel):
    user_id: str
    k: int | None = None


class Recommendation(BaseModel):
    item_id: str
    title: str
    reason: str


class RunResponse(BaseModel):
    recommendations: list[Recommendation] = []


def create_app(
    catalog: list[Item] | None = None,
    profiles: dict[str, Profile] | None = None,
    llm: BaseChatModel | None = None,
) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build any deps not already injected (production path loads JSON).
        if not hasattr(app.state, "graph"):
            app.state.catalog = catalog or load_catalog()
            app.state.profiles = profiles or load_profiles()
            app.state.llm = llm or build_llm(settings)
            app.state.graph = build_recommend_graph(app.state.llm, settings)
        yield

    app = FastAPI(title="langgraph 09-recommendations", lifespan=lifespan)

    # When deps are injected (tests), build eagerly so requests work even if the
    # lifespan startup hasn't run (e.g. TestClient used without a context manager).
    if llm is not None:
        app.state.catalog = catalog if catalog is not None else load_catalog()
        app.state.profiles = profiles if profiles is not None else load_profiles()
        app.state.llm = llm
        app.state.graph = build_recommend_graph(llm, settings)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "approach": APPROACH, "usecase": USECASE}

    @app.post("/run", response_model=RunResponse)
    def run(req: RunRequest) -> RunResponse:
        k = req.k if req.k is not None else settings.rec_top_k
        profile = app.state.profiles.get(req.user_id)
        if profile is None:
            raise HTTPException(
                status_code=404, detail=f"unknown user_id: {req.user_id}"
            )
        out = app.state.graph.invoke(
            {
                "profile": profile,
                "catalog": app.state.catalog,
                "k": k,
                "ranked": [],
                "recommendations": [],
            }
        )
        return RunResponse(
            recommendations=[Recommendation(**r) for r in out["recommendations"]]
        )

    return app


app = create_app()
