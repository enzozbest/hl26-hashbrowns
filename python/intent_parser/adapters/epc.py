"""
EPC Adapter — SKELETON
======================
Queries the Energy Performance Certificate (EPC) registry for energy ratings
of buildings in the target area.

Data source:
    https://epc.opendatacommunities.org/
    - Free API, requires signup for an API key
    - Returns energy ratings per certificate (A–G), floor area, heating type,
      estimated energy costs, recommendations
    - Query by postcode, local authority, or certificate reference
    - Rate-limited to 5,000 requests/day on the free tier

Useful for:
    - Checking current EPC ratings in an area (are buildings energy-efficient?)
    - Understanding refurbishment needs for change-of-use projects
    - Filtering by min EPC rating when user has energy constraints

Environment variables:
    EPC_API_TOKEN — your API key from epc.opendatacommunities.org

Quick setup:
    1. Sign up at https://epc.opendatacommunities.org/login
    2. Get your API key from your account page
    3. Set EPC_API_TOKEN=<your key>
    4. Fill in build_queries, execute, and normalize_results below
"""

from __future__ import annotations

import os
from typing import Any

from ..schema import ParsedIntent
from .base import DataSourceAdapter


class EPCAdapter(DataSourceAdapter):
    """Adapter for the EPC Open Data Communities API."""

    def __init__(self, api_token: str | None = None) -> None:
        self.api_token = api_token or os.getenv("EPC_API_TOKEN")
        self.base_url = "https://epc.opendatacommunities.org/api/v1"

    def name(self) -> str:
        return "EPC Energy Ratings"

    def can_handle(self, intent: ParsedIntent) -> bool:
        """EPC data is relevant for residential development or EPC constraints."""
        # Relevant if user has EPC-related constraints.
        for c in intent.constraints:
            if c.category in ("epc_rating", "energy"):
                return True
        # Relevant for residential projects (likely need EPC data).
        if intent.development.category in ("residential", "change_of_use", "home_improvement"):
            return True
        return False

    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build EPC API query payloads.

        TODO: Implement once you have the API key.  The EPC API supports:
            - GET /domestic/search?postcode=SW1A+1AA
            - GET /domestic/search?local-authority=E09000001
            - GET /non-domestic/search?postcode=...

        You'll want to:
            1. Map intent.location.resolved_councils to local-authority codes
            2. Or use postcode if we have a precise address
            3. Filter by EPC rating if intent.constraints has min_epc_rating
        """
        # TODO: Build real queries.
        return []

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute queries against the EPC API.

        TODO: Implement with httpx.  Example::

            import httpx

            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.api_token}:'.encode()).decode()}",
                "Accept": "application/json",
            }
            async with httpx.AsyncClient() as client:
                for query in queries:
                    resp = await client.get(
                        f"{self.base_url}{query['endpoint']}",
                        params=query['params'],
                        headers=headers,
                    )
                    ...
        """
        return []

    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Normalize EPC API responses into common result format.

        TODO: Map EPC fields to our common format::

            {
                "source": "epc",
                "type": "energy_certificate",
                "title": "EPC Rating B — 14 Example Street",
                "description": "Floor area: 85sqm, Current rating: B (82), ...",
                "location": {"lat": ..., "lng": ...},
                "council": "Hackney",
                "decision": None,
                "date": "2023-05-15",  # lodgement date
                "relevance_score": 0.8,
                "raw": { ... },
            }
        """
        return []
