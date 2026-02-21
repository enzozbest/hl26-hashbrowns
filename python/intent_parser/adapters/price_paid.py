"""
Price Paid Adapter — SKELETON
==============================
Queries HM Land Registry Price Paid Data for property transaction history
in the target area.

Data source:
    https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
    - Free, no API key needed
    - ~5GB CSV of every property transaction in England & Wales since 1995
    - Updated monthly
    - Fields: price, date, postcode, property type, new/old, duration
      (freehold/leasehold), address, local authority

Recommended setup:
    1. Download the complete CSV from gov.uk (or just the yearly files)
    2. Load into SQLite for fast querying:
       CREATE TABLE price_paid (
           id TEXT PRIMARY KEY,
           price INTEGER,
           date TEXT,
           postcode TEXT,
           property_type TEXT,  -- D=detached, S=semi, T=terraced, F=flat, O=other
           new_build TEXT,      -- Y or N
           duration TEXT,       -- F=freehold, L=leasehold
           paon TEXT,           -- primary addressable object
           saon TEXT,           -- secondary addressable object
           street TEXT,
           locality TEXT,
           town TEXT,
           district TEXT,
           county TEXT,
           record_type TEXT     -- A=add, C=change, D=delete
       );
       CREATE INDEX idx_postcode ON price_paid(postcode);
       CREATE INDEX idx_district ON price_paid(district);
    3. Point PRICE_PAID_DB env var to the SQLite file

Useful for:
    - Showing average property prices in target area
    - Comparing areas by price trends
    - Estimating land values for feasibility studies
    - Understanding whether "affordable" makes sense for the area

Environment variables:
    PRICE_PAID_DB — path to SQLite database file
"""

from __future__ import annotations

import os
from typing import Any

from ..schema import ParsedIntent
from .base import DataSourceAdapter


class PricePaidAdapter(DataSourceAdapter):
    """Adapter for HM Land Registry Price Paid Data."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("PRICE_PAID_DB")

    def name(self) -> str:
        return "HM Land Registry Price Paid"

    def can_handle(self, intent: ParsedIntent) -> bool:
        """Price data is relevant for valuation, comparison, or feasibility."""
        # Check constraints for budget mentions.
        for c in intent.constraints:
            if c.category in ("budget", "price", "valuation", "affordability"):
                return True
        # Check analysis goals.
        for g in intent.analysis_goals:
            if g.goal in ("compare_areas", "check_feasibility", "track_trends"):
                return True
        # Relevant for residential if comparing areas.
        if intent.development.category == "residential" and len(intent.location.resolved_councils) > 1:
            return True
        return False

    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build price paid lookup queries.

        TODO: Implement.  Query the SQLite database::

            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                \"\"\"
                SELECT AVG(price), COUNT(*), property_type, strftime('%Y', date) as year
                FROM price_paid
                WHERE district = ?
                  AND date >= ?
                GROUP BY property_type, year
                ORDER BY year DESC
                \"\"\",
                (council_name, date_from),
            )

        You'll want to:
            1. Map intent.location.resolved_councils to district names
            2. Filter by date range from analysis_goals
            3. Aggregate by property type and year for trends
            4. Optionally filter by property_type based on development category
               (F=flat for residential, etc.)
        """
        # TODO: Build real queries.
        return []

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute price paid queries against the SQLite database.

        TODO: Implement using aiosqlite or run sqlite3 in a thread executor::

            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query['sql'], query['params']) as cursor:
                    rows = await cursor.fetchall()
                    ...
        """
        return []

    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Normalize price paid data into common result format.

        TODO: Map aggregated price data to our common format::

            {
                "source": "price_paid",
                "type": "price_summary",
                "title": "Hackney — Avg flat price £425,000 (2024)",
                "description": "Based on 342 transactions. Trend: +5.2% YoY",
                "location": {"lat": ..., "lng": ...},
                "council": "Hackney",
                "decision": None,
                "date": "2024-12-31",
                "relevance_score": 0.7,
                "raw": { "avg_price": 425000, "count": 342, ... },
            }
        """
        return []
