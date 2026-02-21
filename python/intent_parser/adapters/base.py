"""Abstract adapter interface for external data sources.

Each data source (IBex, EPC register, flood-risk service, etc.) gets its own
adapter subclass.  The adapter's job is three-fold:

1. **Transform** a ``ParsedIntent`` into API-specific query parameters.
2. **Execute** the query against the real API.
3. **Normalise** the response into a common result format.

The ``search`` method chains all three steps into a single call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schema import ParsedIntent


class DataSourceAdapter(ABC):
    """Base class for all data source adapters.

    Each adapter knows how to translate our canonical ``ParsedIntent`` into
    one or more API-specific request payloads, execute them, and normalise
    the responses back into a common format.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this data source."""
        ...

    @abstractmethod
    def can_handle(self, intent: ParsedIntent) -> bool:
        """Return ``True`` if this adapter can contribute data for *intent*."""
        ...

    @abstractmethod
    def build_queries(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Transform a ``ParsedIntent`` into API-specific query payloads.

        Returns a list of dicts — each dict is a raw query payload for this
        API.  This is the adapter's **main job**: mapping domain concepts to
        API parameters.
        """
        ...

    @abstractmethod
    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute *queries* against the actual API.

        Returns raw API responses (one per query).
        """
        ...

    @abstractmethod
    def normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[dict[str, Any]]:
        """Transform API-specific responses into our common result format.

        Each item in the returned list should have the shape::

            {
                "source": "ibex",
                "type": "planning_application",
                "title": str,
                "description": str,
                "location": {"lat": float, "lng": float},
                "council": str,
                "decision": str | None,
                "date": str | None,
                "relevance_score": float,   # 0–1
                "raw": dict,                # original API response item
            }
        """
        ...

    # ------------------------------------------------------------------
    # Convenience pipeline
    # ------------------------------------------------------------------

    async def search(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Full pipeline: build queries → execute → normalise.

        Returns an empty list if :meth:`can_handle` is ``False``.
        """
        if not self.can_handle(intent):
            return []
        queries = self.build_queries(intent)
        raw = await self.execute(queries)
        return self.normalize_results(raw, intent)
