# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Raghuveer Dendukuri
# Author: Raghuveer Dendukuri · Co-author: Claude Code (Opus)
# ai-usecases — UC09 Personalised recommendations (claude-agent-sdk). See claude-agent-sdk/09-recommendations/README.md
"""Personalised recommendations: profile + catalog tools, agent-ranked, explained.

The agent cannot see the profile or the catalog in its prompt — it must fetch
both through tools, then emit a ranked shortlist with a per-item reason. Keeping
the data out of the prompt is what makes the explanations checkable: every
recommended id is validated against the real catalog, so an invented product is
caught here rather than shipped to a user.

    get_profile ──► list_catalog ──► emit_recommendations(items, rationale)

**Fit note.** Ranking a small catalog does not need an agent loop; a single
prompt with the catalog inlined would be cheaper. What the agent buys is the
*explanation* being grounded in fetched data rather than in the prompt, and the
ability to consult only the categories the profile implies. On a catalog of five
items that is a modest win — see the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import BaseModel, Field, ValidationError

from .agent import Runner, build_options, default_runner
from .settings import Settings

PROFILES: dict[str, dict[str, Any]] = {
    "u-1": {
        "name": "Ada",
        "likes": ["office", "productivity"],
        "dislikes": ["kitchen"],
        "recent_purchases": ["Mechanical Keyboard"],
        "budget_usd": 450,
    },
    "u-2": {
        "name": "Grace",
        "likes": ["home", "kitchen"],
        "dislikes": [],
        "recent_purchases": ["Robot Vacuum"],
        "budget_usd": 600,
    },
}

CATALOG: list[dict[str, Any]] = [
    {"id": "p-1", "name": "Robot Vacuum", "category": "home", "price": 249.00},
    {"id": "p-2", "name": "Standing Desk", "category": "office", "price": 399.50},
    {"id": "p-3", "name": "Mechanical Keyboard", "category": "office", "price": 129.99},
    {"id": "p-4", "name": "Espresso Machine", "category": "kitchen", "price": 549.00},
    {"id": "p-5", "name": "Desk Lamp", "category": "office", "price": 39.95},
    {"id": "p-6", "name": "Monitor Arm", "category": "office", "price": 89.00},
    {"id": "p-7", "name": "Air Fryer", "category": "kitchen", "price": 119.00},
]

CATALOG_IDS = {item["id"] for item in CATALOG}


class Recommendation(BaseModel):
    id: str = Field(max_length=40)
    reason: str = Field(max_length=400)


class Recommendations(BaseModel):
    items: list[Recommendation] = Field(min_length=1, max_length=5)
    rationale: str = Field(max_length=1000)


RECOMMEND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Ranked recommendations, best first. 1-5 items.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Catalog product id, e.g. p-2."},
                    "reason": {
                        "type": "string",
                        "description": "Why this suits THIS user, referencing their profile.",
                    },
                },
                "required": ["id", "reason"],
            },
        },
        "rationale": {
            "type": "string",
            "description": "One short paragraph on the overall strategy.",
        },
    },
    "required": ["items", "rationale"],
}

PROFILE_TOOL = "mcp__reco__get_profile"
CATALOG_TOOL = "mcp__reco__list_catalog"
EMIT_TOOL = "mcp__reco__emit_recommendations"


@tool("get_profile", "Fetch a user's preference profile by id, e.g. u-1.", {"user_id": str})
async def get_profile(args: dict[str, Any]) -> dict[str, Any]:
    profile = PROFILES.get(str(args.get("user_id", "")).strip().lower())
    if profile is None:
        return {
            "content": [{"type": "text", "text": "No such user."}],
            "is_error": True,
        }
    likes = ", ".join(profile["likes"]) or "none"
    dislikes = ", ".join(profile["dislikes"]) or "none"
    bought = ", ".join(profile["recent_purchases"]) or "none"
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"{profile['name']} — likes: {likes}; dislikes: {dislikes}; "
                    f"recently bought: {bought}; budget: ${profile['budget_usd']}"
                ),
            }
        ]
    }


@tool(
    "list_catalog",
    "List catalog products, optionally filtered to one category.",
    {"category": str},
)
async def list_catalog(args: dict[str, Any]) -> dict[str, Any]:
    category = str(args.get("category", "")).strip().lower()
    items = [i for i in CATALOG if not category or i["category"] == category]
    if not items:
        return {
            "content": [{"type": "text", "text": f"No products in category '{category}'."}],
            "is_error": True,
        }
    lines = [f"{i['id']} | {i['name']} | {i['category']} | ${i['price']:.2f}" for i in items]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "emit_recommendations",
    "Record the final ranked recommendations. Call exactly once, last.",
    RECOMMEND_SCHEMA,
)
async def emit_recommendations(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": "Recorded."}]}


SYSTEM_PROMPT = """You recommend products to a specific user.

Steps:
1. Call get_profile for the user id you are given.
2. Call list_catalog (optionally per category) to see what is available.
3. Call emit_recommendations exactly once with up to 5 ranked items.

Rules:
- Only recommend ids that appeared in list_catalog output. Never invent one.
- Do not recommend something the user recently bought, or in a disliked category.
- Respect the user's budget for any single item.
- Each reason must refer to something specific from their profile."""

RECO_TOOLS = [PROFILE_TOOL, CATALOG_TOOL, EMIT_TOOL]


def build_reco_server():
    return create_sdk_mcp_server(
        name="reco",
        version="1.0.0",
        tools=[get_profile, list_catalog, emit_recommendations],
    )


@dataclass
class RecoResult:
    valid: bool
    items: list[dict[str, Any]]
    rationale: str
    errors: list[str]
    num_turns: int
    cost_usd: float


async def recommend(
    user_id: str, settings: Settings, runner: Runner | None = None
) -> RecoResult:
    runner = runner or default_runner
    options = build_options(
        settings,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=RECO_TOOLS,
        mcp_servers={"reco": build_reco_server()},
        max_turns=max(settings.agent_max_turns, 6),
    )
    result = await runner(f"Recommend products for user {user_id}.", options)

    emits = [c for c in result.tool_calls if c.name == EMIT_TOOL]
    if not emits:
        return RecoResult(
            valid=False,
            items=[],
            rationale="",
            errors=["agent did not call emit_recommendations"],
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
        )

    try:
        parsed = Recommendations.model_validate(emits[-1].input)
    except ValidationError as exc:
        return RecoResult(
            valid=False,
            items=[],
            rationale="",
            errors=[
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ],
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
        )

    # Ground the output: every id must exist in the real catalog. A hallucinated
    # product is caught here, not shipped to a user.
    unknown = [item.id for item in parsed.items if item.id not in CATALOG_IDS]
    if unknown:
        return RecoResult(
            valid=False,
            items=[],
            rationale="",
            errors=[f"recommended ids not in catalog: {', '.join(unknown)}"],
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
        )

    by_id = {i["id"]: i for i in CATALOG}
    items = [
        {
            "id": item.id,
            "name": by_id[item.id]["name"],
            "category": by_id[item.id]["category"],
            "price": by_id[item.id]["price"],
            "reason": item.reason,
        }
        for item in parsed.items
    ]
    return RecoResult(
        valid=True,
        items=items,
        rationale=parsed.rationale,
        errors=[],
        num_turns=result.num_turns,
        cost_usd=result.cost_usd,
    )
