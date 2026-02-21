"""Search orchestrator — fans out a ParsedIntent to all registered adapters.

The orchestrator is the brain of the search pipeline.  It takes a canonical
``ParsedIntent``, runs every applicable adapter concurrently, and merges
the results into a single structured response.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from intent_parser.adapters.base import DataSourceAdapter
from intent_parser.schema import ParsedIntent


class SearchOrchestrator:
    """Registry + runner for data-source adapters."""

    def __init__(self) -> None:
        self.adapters: list[DataSourceAdapter] = []

    def register(self, adapter: DataSourceAdapter) -> None:
        """Add an adapter to the registry."""
        self.adapters.append(adapter)

    # ------------------------------------------------------------------
    # Full search
    # ------------------------------------------------------------------

    async def search(self, intent: ParsedIntent) -> dict[str, Any]:
        """Run all applicable adapters concurrently and merge results.

        Returns a dict with the shape::

            {
                "intent": { ... },
                "intent_summary": "...",
                "results_by_source": {
                    "ibex": [...],
                    ...
                },
                "merged_results": [...],
                "metadata": {
                    "sources_queried": ["ibex", ...],
                    "total_results": 42,
                    "query_time_ms": 1200,
                },
            }
        """
        applicable = [a for a in self.adapters if a.can_handle(intent)]

        t0 = time.monotonic()
        tasks = [a.search(intent) for a in applicable]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        results_by_source: dict[str, list[dict[str, Any]]] = {}
        sources_queried: list[str] = []

        for adapter, outcome in zip(applicable, settled):
            name = adapter.name()
            sources_queried.append(name)
            if isinstance(outcome, BaseException):
                results_by_source[name] = [{
                    "error": str(outcome),
                    "type": type(outcome).__name__,
                }]
            else:
                results_by_source[name] = outcome

        # Merge all non-error results and sort by relevance.
        merged: list[dict[str, Any]] = []
        for items in results_by_source.values():
            for item in items:
                if "error" not in item:
                    merged.append(item)
        merged.sort(key=lambda r: r.get("relevance_score", 0), reverse=True)

        return {
            "intent": intent.to_dict(),
            "intent_summary": intent.to_summary(),
            "results_by_source": results_by_source,
            "merged_results": merged,
            "metadata": {
                "sources_queried": sources_queried,
                "total_results": len(merged),
                "query_time_ms": elapsed_ms,
            },
        }

    # ------------------------------------------------------------------
    # Dry-run query plan
    # ------------------------------------------------------------------

    def get_query_plans(self, intent: ParsedIntent) -> dict[str, Any]:
        """Show what queries *would* be made without executing them.

        Useful for debugging and for the frontend to display
        "here's what we understood, and here's what we'd search for".
        """
        plans: dict[str, list[dict[str, Any]]] = {}
        for adapter in self.adapters:
            if adapter.can_handle(intent):
                queries = adapter.build_queries(intent)
                plans[adapter.name()] = [
                    {
                        "endpoint": q.get("_endpoint", "unknown"),
                        "description": q.get("_description", ""),
                        "payload": q.get("payload", q),
                    }
                    for q in queries
                ]
        return plans
