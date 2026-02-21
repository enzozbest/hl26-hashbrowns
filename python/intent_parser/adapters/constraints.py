"""
Planning Constraints Adapter — SKELETON
========================================
Checks whether target locations fall within conservation areas, green belt,
Article 4 direction areas, or listed building zones using open planning data.

Data source:
    https://www.planning.data.gov.uk/
    - Free, no API key needed
    - GeoJSON downloads for:
        * Conservation areas
        * Green belt boundaries
        * Article 4 direction areas
        * Listed buildings (point data)
        * Tree preservation orders
        * Areas of Outstanding Natural Beauty (AONB)
    - Also has a live API: https://www.planning.data.gov.uk/docs

Recommended setup:
    1. Download relevant GeoJSON datasets from planning.data.gov.uk:
       - conservation-area.geojson
       - green-belt.geojson
       - article-4-direction-area.geojson
    2. Place in data/constraints/ directory
    3. pip install shapely
    4. Load polygons, build R-tree spatial index, do point-in-polygon checks

    Alternatively, use the live API:
        GET https://www.planning.data.gov.uk/api/v1/entity.geojson
            ?dataset=conservation-area
            &longitude=-0.05&latitude=51.55
            &limit=10

Useful for:
    - Checking if a target site falls in a conservation area (stricter design rules)
    - Green belt checks (very hard to get permission for new build)
    - Article 4 directions (removes permitted development rights, e.g. no
      automatic office-to-residential conversion)
    - Listed building proximity (affects what you can build nearby)

Environment variables:
    CONSTRAINTS_DATA_DIR — path to directory containing GeoJSON files
"""

from __future__ import annotations

import os
from typing import Any

from ..schema import ParsedIntent
from .base import DataSourceAdapter


class ConstraintsAdapter(DataSourceAdapter):
    """Adapter for planning constraint boundary data (conservation areas,
    green belt, Article 4 directions, listed buildings)."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = data_dir or os.getenv("CONSTRAINTS_DATA_DIR")

    def name(self) -> str:
        return "Planning Constraints"

    def can_handle(self, intent: ParsedIntent) -> bool:
        """Constraints data is relevant if user mentions any designations."""
        relevant_categories = {
            "conservation_area", "green_belt", "article_4",
            "listed_building", "aonb", "tree_preservation",
        }
        for c in intent.constraints:
            if c.category in relevant_categories:
                return True
        return False

    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build constraint check queries.

        TODO: Implement.  Two approaches:

        Approach A — local GeoJSON point-in-polygon::

            from shapely.geometry import Point, shape
            from shapely.strtree import STRtree
            import json

            # Load once at startup
            with open("data/constraints/conservation-area.geojson") as f:
                conservation = json.load(f)
            polygons = [shape(f["geometry"]) for f in conservation["features"]]
            tree = STRtree(polygons)

            # Query
            point = Point(lng, lat)
            hits = tree.query(point)
            in_conservation = any(polygons[i].contains(point) for i in hits)

        Approach B — planning.data.gov.uk API::

            GET https://www.planning.data.gov.uk/api/v1/entity.geojson
                ?dataset=conservation-area
                &longitude={lng}&latitude={lat}
                &limit=10

        You'll want to:
            1. Get coordinates from intent.location.resolved_coordinates
            2. Check each relevant constraint type the user mentioned
            3. Return which designations the location falls within
        """
        # TODO: Build real queries.
        return []

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute constraint boundary queries.

        TODO: Implement — either local GeoJSON lookups or API calls.
        Local GeoJSON with shapely is recommended for speed and offline use.
        """
        return []

    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Normalize constraint check results into common format.

        TODO: Map constraint data to our common format::

            {
                "source": "planning_constraints",
                "type": "constraint_check",
                "title": "Conservation Area — Hackney Downs",
                "description": "Location falls within the Hackney Downs conservation area. "
                               "Stricter design and materials requirements apply.",
                "location": {"lat": ..., "lng": ...},
                "council": "Hackney",
                "decision": None,
                "date": None,
                "relevance_score": 0.95,  # high if user wanted to avoid this
                "raw": {
                    "designation": "conservation_area",
                    "name": "Hackney Downs",
                    "in_zone": True,
                    ...
                },
            }
        """
        return []
