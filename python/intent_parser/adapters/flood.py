"""
Flood Risk Adapter — SKELETON
==============================
Checks whether target locations fall within flood risk zones, using
Environment Agency data.

Data sources:
    Option A — Live API:
        https://environment.data.gov.uk/flood-monitoring/id/floods
        - Free, no API key needed
        - Real-time flood warnings and flood area geometries
        - Good for current status but less useful for planning zone data

    Option B — Downloaded GeoJSON (recommended for hackathon):
        https://flood-map-for-planning.service.gov.uk/
        - Download flood zone 2 and 3 shapefiles/GeoJSON
        - Do point-in-polygon checks locally using shapely
        - Much faster and no rate limits

    Flood zones:
        Zone 1 = low risk (<0.1% annual probability)
        Zone 2 = medium risk (0.1–1% river, 0.1–0.5% sea)
        Zone 3a = high risk (>1% river, >0.5% sea)
        Zone 3b = functional floodplain

Useful for:
    - Checking if a target site is in a flood zone
    - Filtering out high-risk areas when user says "avoid flood zones"
    - Risk assessment overlay for site comparison

Quick setup (GeoJSON approach):
    1. Download flood zone GeoJSON from flood-map-for-planning.service.gov.uk
    2. Place in data/flood_zones/ directory
    3. pip install shapely
    4. Load GeoJSON, build spatial index, do point-in-polygon checks
"""

from __future__ import annotations

from typing import Any

from ..schema import ParsedIntent
from .base import DataSourceAdapter


class FloodRiskAdapter(DataSourceAdapter):
    """Adapter for flood risk zone data (Environment Agency)."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = data_dir  # Path to downloaded GeoJSON files.

    def name(self) -> str:
        return "Flood Risk Zones"

    def can_handle(self, intent: ParsedIntent) -> bool:
        """Flood data is relevant if user mentions flood risk or wants risk assessment."""
        # Check constraints.
        for c in intent.constraints:
            if c.category in ("flood_risk", "flood_zone", "flood"):
                return True
        # Check analysis goals.
        for g in intent.analysis_goals:
            if g.goal == "assess_risk":
                return True
        return False

    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build flood risk check queries.

        TODO: Implement.  Two approaches:

        Approach A — local GeoJSON point-in-polygon::

            from shapely.geometry import Point, shape
            import json

            with open("data/flood_zones/flood_zone_3.geojson") as f:
                zones = json.load(f)
            point = Point(lng, lat)
            for feature in zones["features"]:
                polygon = shape(feature["geometry"])
                if polygon.contains(point):
                    return zone info

        Approach B — Environment Agency API::

            GET https://environment.data.gov.uk/flood-monitoring/id/floods
            ?lat={lat}&long={lng}&dist=5

        You'll want to:
            1. Get coordinates from intent.location.resolved_coordinates
            2. Check each coordinate against flood zone polygons
            3. Return the flood zone classification (1, 2, 3a, 3b)
        """
        # TODO: Build real queries.
        return []

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute flood risk queries.

        TODO: Implement — either local GeoJSON lookups or API calls.
        Local GeoJSON is recommended as it's faster and works offline.
        """
        return []

    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Normalize flood risk results into common format.

        TODO: Map flood zone data to our common format::

            {
                "source": "flood_risk",
                "type": "flood_zone_check",
                "title": "Flood Zone 2 — Medium Risk",
                "description": "Location is within Flood Zone 2 (0.1–1% annual probability)",
                "location": {"lat": ..., "lng": ...},
                "council": "Lambeth",
                "decision": None,
                "date": None,
                "relevance_score": 0.9,  # high relevance if user asked about flood risk
                "raw": { "zone": 2, "risk_level": "medium", ... },
            }
        """
        return []
