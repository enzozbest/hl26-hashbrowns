"""FastAPI application for the planning site finder.

Endpoints:
    POST /api/parse    — Parse a query into a structured ParsedIntent
    POST /api/plan     — Parse + show what API queries would be made (dry run)
    POST /api/search   — Full pipeline: parse → orchestrate → return results
    GET  /api/councils — Council lookup for frontend autocomplete
    GET  /api/adapters — Registered adapters and their status
    GET  /api/health   — Health check

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from intent_parser.adapters.constraints import ConstraintsAdapter
from intent_parser.adapters.epc import EPCAdapter
from intent_parser.adapters.flood import FloodRiskAdapter
from intent_parser.adapters.ibex import IBexAdapter
from intent_parser.adapters.price_paid import PricePaidAdapter
from intent_parser.llm_parser import IntentParseError, parse_query, parse_query_mock
from intent_parser.location import UK_COUNCILS
from intent_parser.schema import ParsedIntent

from .orchestrator import SearchOrchestrator

# ---------------------------------------------------------------------------
# Globals wired up at startup
# ---------------------------------------------------------------------------

orchestrator = SearchOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register all data-source adapters at startup.

    Each adapter gracefully degrades to returning empty results when its
    credentials / data files are not configured.
    """
    orchestrator.register(IBexAdapter(
        api_token=os.getenv("IBEX_API_TOKEN"),
        base_url=os.getenv("IBEX_BASE_URL", "https://api.ibexenterprise.com"),
    ))
    orchestrator.register(EPCAdapter(
        api_token=os.getenv("EPC_API_TOKEN"),
    ))
    orchestrator.register(FloodRiskAdapter(
        data_dir=os.getenv("FLOOD_DATA_DIR"),
    ))
    orchestrator.register(PricePaidAdapter(
        db_path=os.getenv("PRICE_PAID_DB"),
    ))
    orchestrator.register(ConstraintsAdapter(
        data_dir=os.getenv("CONSTRAINTS_DATA_DIR"),
    ))
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Planning Site Finder",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(query: str) -> ParsedIntent:
    """Parse a query, using the real LLM if GOOGLE_API_KEY is set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return await parse_query(query, api_key=api_key)
    return parse_query_mock(query)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/parse")
async def parse_endpoint(body: QueryRequest) -> dict[str, Any]:
    """Parse a natural language query into a structured ParsedIntent.

    This is the most important endpoint — the frontend calls this first.
    """
    try:
        intent = await _parse(body.query)
    except IntentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "intent": intent.to_dict(),
        "summary": intent.to_summary(),
    }


@app.post("/api/plan")
async def plan_endpoint(body: QueryRequest) -> dict[str, Any]:
    """Parse intent + show what API queries *would* be made (dry run).

    No actual API calls are made.  Useful before you have API keys, and
    for the frontend to display "here's what we'd search for".
    """
    try:
        intent = await _parse(body.query)
    except IntentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    plans = orchestrator.get_query_plans(intent)
    return {
        "intent": intent.to_dict(),
        "summary": intent.to_summary(),
        "query_plans": plans,
    }


@app.post("/api/search")
async def search_endpoint(body: QueryRequest) -> dict[str, Any]:
    """Full pipeline: parse → orchestrate across all adapters → return results."""
    try:
        intent = await _parse(body.query)
    except IntentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        results = await orchestrator.search(intent)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}")
    return results


@app.get("/api/councils")
async def councils_endpoint() -> dict[str, Any]:
    """Return the council lookup for frontend autocomplete.

    Each entry includes name, coordinates, region, and aliases.
    """
    councils = []
    for name, data in UK_COUNCILS.items():
        councils.append({
            "name": name,
            "lat": data["lat"],
            "lng": data["lng"],
            "region": data["region"],
            "sub_region": data["sub_region"],
            "country": data["country"],
            "aliases": data.get("aliases", []),
        })
    return {"councils": councils, "count": len(councils)}


@app.get("/api/adapters")
async def adapters_endpoint() -> dict[str, Any]:
    """Return registered adapters and whether they have credentials configured."""
    adapters = []
    for adapter in orchestrator.adapters:
        info: dict[str, Any] = {"name": adapter.name(), "mode": "skeleton"}
        if isinstance(adapter, IBexAdapter):
            info["has_api_token"] = adapter.api_token is not None
            info["base_url"] = adapter.base_url
            info["mode"] = "live" if adapter.api_token else "mock"
        elif isinstance(adapter, EPCAdapter):
            info["has_api_token"] = adapter.api_token is not None
            info["mode"] = "live" if adapter.api_token else "skeleton"
        elif isinstance(adapter, FloodRiskAdapter):
            info["has_data"] = adapter.data_dir is not None
            info["mode"] = "live" if adapter.data_dir else "skeleton"
        elif isinstance(adapter, PricePaidAdapter):
            info["has_data"] = adapter.db_path is not None
            info["mode"] = "live" if adapter.db_path else "skeleton"
        elif isinstance(adapter, ConstraintsAdapter):
            info["has_data"] = adapter.data_dir is not None
            info["mode"] = "live" if adapter.data_dir else "skeleton"
        adapters.append(info)
    return {"adapters": adapters}


@app.get("/api/health")
async def health_endpoint() -> dict[str, str]:
    return {"status": "ok"}
