"""
IBex Adapter — SKELETON
=======================
This adapter is intentionally incomplete.  Fill in the TODOs once you:

1. Have your IBex API key from the hackathon
2. Make a test call and see what the response looks like
3. Know the exact filter values (normalised_application_type options,
   project_type options, council_id mappings)

The adapter pattern means you ONLY need to edit this file.
The intent parser, API endpoints, and frontend don't need to change.

Quick checklist when you get the API key:
□ Test GET/POST to find the real base URL and endpoints
□ Log a raw response and paste it into normalize_results to build the mapping
□ Create a COUNCIL_ID_MAP dict mapping council names → IBex integer IDs
□ Map development categories to their normalised_application_type values
□ Map development categories to their project_type values
"""

from __future__ import annotations

import os
from typing import Any

from ..schema import ParsedIntent
from .base import DataSourceAdapter


# ---------------------------------------------------------------------------
# Council name → IBex integer ID mapping
# ---------------------------------------------------------------------------
# TODO: Populate this once you can query the IBex API for council IDs.
# Example:
#   "Hackney": 42,
#   "Lambeth": 87,
COUNCIL_ID_MAP: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Development category → IBex filter value mappings
# ---------------------------------------------------------------------------
# TODO: Confirm exact string values from a real API response.

CATEGORY_TO_APPLICATION_TYPE: dict[str, list[str]] = {
    "residential": ["full planning application", "outline planning application"],
    "commercial": ["full planning application"],
    "mixed_use": ["full planning application"],
    "home_improvement": ["householder"],
    "change_of_use": ["change of use"],
    "infrastructure": ["full planning application"],
    "hospitality": ["full planning application"],
    "industrial": ["full planning application"],
    "retail": ["full planning application"],
    "demolition": ["full planning application"],
}

CATEGORY_TO_PROJECT_TYPE: dict[str, list[str]] = {
    "residential": ["new build"],
    "commercial": ["new build"],
    "mixed_use": ["new build"],
    "home_improvement": ["home improvement"],
    "change_of_use": ["conversion"],
    "infrastructure": ["infrastructure"],
    "hospitality": ["new build", "conversion"],
    "industrial": ["new build"],
    "retail": ["new build"],
    "demolition": ["demolition"],
}


class IBexAdapter(DataSourceAdapter):
    """Adapter for the IBex planning data API.

    Translates a ``ParsedIntent`` into IBex ``/search`` (location-based) and
    ``/applications`` (attribute-based) queries.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_token = api_token or os.getenv("IBEX_API_TOKEN")
        self.base_url = (base_url or "https://api.ibexenterprise.com").rstrip("/")

    def name(self) -> str:
        return "IBex Planning Data"

    def can_handle(self, intent: ParsedIntent) -> bool:
        # IBex can handle anything with a resolvable location.
        return True

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build IBex API payloads from a ``ParsedIntent``.

        Produces up to two queries:

        1. ``/search`` — location-based search (if we have coordinates).
        2. ``/applications`` — attribute-based search (if we have keywords
           or date ranges).
        """
        queries: list[dict[str, Any]] = []

        # --- Query 1: location-based search --------------------------------
        if intent.location.resolved_coordinates:
            coords = intent.location.resolved_coordinates
            easting, northing = self._to_osgb36(coords["lat"], coords["lng"])

            filters: dict[str, Any] = {}
            app_types = CATEGORY_TO_APPLICATION_TYPE.get(intent.development.category)
            if app_types:
                filters["normalised_application_type"] = app_types
            proj_types = CATEGORY_TO_PROJECT_TYPE.get(intent.development.category)
            if proj_types:
                filters["project_type"] = proj_types

            queries.append({
                "_endpoint": "/search",
                "_description": "Search by location radius",
                "payload": {
                    "input": {
                        "coordinates": [easting, northing],
                        "radius": intent.location.radius_suggestion_m or 1000,
                        "srid": 27700,
                    },
                    "filters": filters,
                    "extensions": {
                        "appeals": False,
                        "heading": False,
                    },
                },
            })

        # --- Query 2: attribute-based search --------------------------------
        if intent.keywords or intent.analysis_goals:
            time_range = self._get_time_range(intent)

            input_block: dict[str, Any] = {
                "date_from": time_range["from"],
                "date_to": time_range["to"],
            }

            # Map resolved council names → IBex IDs.
            council_ids = self._resolve_council_ids(intent.location.resolved_councils)
            if council_ids:
                input_block["council_id"] = council_ids

            filters_block: dict[str, Any] = {}
            if intent.keywords:
                filters_block["keywords"] = intent.keywords[:5]

            # Decide which decisions to request.
            decisions = self._decisions_for_intent(intent)
            if decisions:
                filters_block["normalised_decision"] = decisions

            queries.append({
                "_endpoint": "/applications",
                "_description": "Search by attributes",
                "payload": {
                    "input": input_block,
                    "filters": filters_block,
                },
            })

        return queries

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute queries against the IBex API.

        Returns mock stubs when no API token is configured.

        TODO: Implement real HTTP calls once you have the token.  Example::

            import httpx

            async with httpx.AsyncClient() as client:
                results = []
                for query in queries:
                    resp = await client.post(
                        f"{self.base_url}{query['_endpoint']}",
                        json=query["payload"],
                        headers={"Authorization": f"Bearer {self.api_token}"},
                    )
                    resp.raise_for_status()
                    results.append(resp.json())
                return results
        """
        if not self.api_token:
            return [
                {"_mock": True, "_query": q, "results": []}
                for q in queries
            ]

        # TODO: Replace with real httpx calls (see docstring above).
        return []

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Transform raw IBex responses into the common result format.

        TODO: Implement once you can see a real response payload.  Each
        result item should be mapped to::

            {
                "source": "ibex",
                "type": "planning_application",
                "title": ...,
                "description": ...,
                "location": {"lat": ..., "lng": ...},
                "council": ...,
                "decision": ...,
                "date": ...,
                "relevance_score": ...,
                "raw": <original item>,
            }
        """
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_osgb36(lat: float, lng: float) -> tuple[float, float]:
        """Convert WGS84 lat/lng to approximate OSGB36 easting/northing.

        This is a rough linear approximation that works for England/Wales.
        For production accuracy, use ``pyproj`` or the OS Grid transformation.

        TODO: Replace with ``pyproj.Transformer`` for sub-metre accuracy::

            from pyproj import Transformer
            tx = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            easting, northing = tx.transform(lng, lat)
        """
        # Linear approximation centred on London.
        easting = 500_000 + (lng + 0.1) * 70_000
        northing = 180_000 + (lat - 51.5) * 111_000
        return (round(easting, 1), round(northing, 1))

    @staticmethod
    def _resolve_council_ids(council_names: list[str]) -> list[int]:
        """Map council names to IBex integer IDs using ``COUNCIL_ID_MAP``."""
        ids = []
        for name in council_names:
            if name in COUNCIL_ID_MAP:
                ids.append(COUNCIL_ID_MAP[name])
        return ids

    @staticmethod
    def _get_time_range(intent: ParsedIntent) -> dict[str, str]:
        """Extract a date range from the intent's analysis goals.

        Falls back to the last 3 years if no goal specifies a range.
        """
        for goal in intent.analysis_goals:
            if goal.time_range:
                return goal.time_range
        # Default: last 3 years.
        from datetime import date, timedelta

        today = date.today()
        three_years_ago = today - timedelta(days=3 * 365)
        return {"from": three_years_ago.isoformat(), "to": today.isoformat()}

    @staticmethod
    def _decisions_for_intent(intent: ParsedIntent) -> list[str]:
        """Determine which decision statuses to request based on the intent."""
        for goal in intent.analysis_goals:
            if goal.goal == "understand_refusals":
                return ["Refused"]
            if goal.goal == "compare_areas":
                return ["Approved", "Refused"]
            if goal.goal == "track_trends":
                return ["Approved", "Refused", "Pending", "Withdrawn"]
        # Default: approvals only.
        return ["Approved"]
