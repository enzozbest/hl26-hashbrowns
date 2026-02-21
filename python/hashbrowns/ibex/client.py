"""IbexClient — public async API for all three Ibex Enterprise API endpoints.

Assembles JWT auth, semaphore concurrency, tenacity retries, pagination, and
413 adaptive subdivision into four clean endpoint methods: search,
search_polygon, applications, and stats.

Usage::

    from hashbrowns.config import Settings
    from hashbrowns.ibex.client import IbexClient

    async with IbexClient(settings) as client:
        results = await client.search([528349, 186246], radius=300, srid=27700)

Retry strategy
--------------
- 5xx responses and network errors (ConnectError, TimeoutException, ReadError)
  trigger tenacity retry: up to 3 attempts, exponential backoff (1s → 10s).
- 403 raises IbexAuthError immediately — not retried.
- 413 from /search triggers automatic radius halving (up to MAX_413_DEPTH=4).
- 413 from /applications or /stats propagates as PayloadTooLargeError.
"""
from __future__ import annotations

import asyncio

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hashbrowns.config import Settings
from hashbrowns.ibex.models import ApplicationsResponse, SearchResponse, StatsResponse
from hashbrowns.ibex.pagination import (
    PayloadTooLargeError,
    paginate_applications,
    paginate_search,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IbexAuthError(Exception):
    """Raised on HTTP 403 — JWT invalid or expired."""


class SearchAreaTooSmallError(Exception):
    """Raised when 413 subdivision reaches minimum radius floor."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class IbexClient:
    """Async HTTP client for the Ibex Enterprise API.

    Provides four endpoint methods (search, search_polygon, applications,
    stats) and implements concurrency limiting, tenacity retries, and
    automatic 413 radius halving for the search endpoint.

    Intended to be used as an async context manager so that the underlying
    httpx.AsyncClient is always properly closed::

        async with IbexClient(settings) as client:
            results = await client.search([528349, 186246], 300, 27700)
    """

    MIN_RADIUS_METRES: float = 50.0
    MAX_413_DEPTH: int = 4

    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ibex_base_url,
            headers={"Authorization": f"Bearer {settings.ibex_api_key}"},
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=30.0),
        )
        self._semaphore = asyncio.Semaphore(settings.ibex_max_concurrency)

    async def __aenter__(self) -> "IbexClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.aclose()

    # -----------------------------------------------------------------------
    # Internal transport layer
    # -----------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.ReadError,
                httpx.HTTPStatusError,
            )
        ),
        reraise=True,
    )
    async def _post(self, path: str, payload: dict) -> httpx.Response:
        """POST *path* with JSON *payload*, returning the raw httpx.Response.

        Handles status codes before delegating to tenacity's retry logic:
        - 403 → raises IbexAuthError immediately (not retried)
        - 413 → raises PayloadTooLargeError immediately (not retried)
        - 5xx → calls response.raise_for_status() which raises
                 httpx.HTTPStatusError, which IS in the tenacity retry set.

        Concurrent calls are gated through self._semaphore.
        """
        async with self._semaphore:
            response = await self._client.post(path, json=payload)

        # Handle status BEFORE raise_for_status to control what tenacity retries
        if response.status_code == 403:
            raise IbexAuthError(f"JWT invalid or expired: {response.text}")
        if response.status_code == 413:
            raise PayloadTooLargeError(response)
        response.raise_for_status()  # 5xx → httpx.HTTPStatusError → tenacity retries
        return response

    # -----------------------------------------------------------------------
    # Public endpoint methods
    # -----------------------------------------------------------------------

    async def search(
        self,
        coordinates: list[float],
        radius: float,
        srid: int,
        *,
        extensions: dict | None = None,
        filters: dict | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        date_range_type: str | None = None,
        _depth: int = 0,
    ) -> list[SearchResponse]:
        """Search for planning applications within a radius of given coordinates.

        On HTTP 413 the radius is automatically halved and the call retried
        (up to MAX_413_DEPTH times). SearchAreaTooSmallError is raised when
        either the radius falls below MIN_RADIUS_METRES or MAX_413_DEPTH
        halvings have been exhausted.

        Args:
            coordinates: [x, y] in the given SRID (e.g. [528349, 186246] for
                BNG SRID 27700 or [lon, lat] for WGS84 SRID 4326).
            radius: Search radius in metres.
            srid: Spatial reference ID — 27700 (BNG) or 4326 (WGS84).
            extensions: Optional dict of extension flags (see OpenAPI spec).
            filters: Optional dict of field filters (see OpenAPI spec).
            date_from: ISO date string, e.g. "2025-01-01".
            date_to: ISO date string, e.g. "2025-03-31".
            date_range_type: "any", "decided", or "validated". Defaults to
                "any" when date_from is provided but date_range_type is None.
            _depth: Internal recursion counter — do not pass externally.

        Returns:
            List of SearchResponse objects from all pages.

        Raises:
            SearchAreaTooSmallError: Radius below minimum or too many halvings.
            IbexAuthError: API returned 403 (invalid / expired JWT).
            httpx.HTTPStatusError: Unrecoverable 5xx after all retries.
        """
        if radius < self.MIN_RADIUS_METRES:
            raise SearchAreaTooSmallError(
                f"Radius {radius}m is below minimum {self.MIN_RADIUS_METRES}m"
            )
        if _depth > self.MAX_413_DEPTH:
            raise SearchAreaTooSmallError(
                f"413 persists after {self.MAX_413_DEPTH} radius halvings"
            )

        payload: dict = {
            "input": {
                "srid": srid,
                "coordinates": coordinates,
                "radius": radius,
            }
        }
        if date_from:
            payload["input"]["date_from"] = date_from
            payload["input"]["date_to"] = date_to
            payload["input"]["date_range_type"] = date_range_type or "any"
        if extensions:
            payload["extensions"] = extensions
        if filters:
            payload["filters"] = filters

        try:
            return await paginate_search(self._post, payload)
        except PayloadTooLargeError:
            return await self.search(
                coordinates,
                radius / 2,
                srid,
                extensions=extensions,
                filters=filters,
                date_from=date_from,
                date_to=date_to,
                date_range_type=date_range_type,
                _depth=_depth + 1,
            )

    async def search_polygon(
        self,
        polygon_coordinates: list[list[list[float]]],
        srid: int,
        *,
        extensions: dict | None = None,
        filters: dict | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        date_range_type: str | None = None,
    ) -> list[SearchResponse]:
        """Search for planning applications within a polygon boundary.

        Args:
            polygon_coordinates: GeoJSON-style polygon ring list, e.g.
                [[[x0, y0], [x1, y1], ..., [x0, y0]]].
            srid: Spatial reference ID — 27700 (BNG) or 4326 (WGS84).
            extensions: Optional dict of extension flags (see OpenAPI spec).
            filters: Optional dict of field filters (see OpenAPI spec).
            date_from: ISO date string, e.g. "2025-01-01".
            date_to: ISO date string, e.g. "2025-03-31".
            date_range_type: "any", "decided", or "validated".

        Returns:
            List of SearchResponse objects from all pages.

        Raises:
            IbexAuthError: API returned 403.
            httpx.HTTPStatusError: Unrecoverable 5xx after all retries.
        """
        payload: dict = {
            "input": {
                "srid": srid,
                "polygon": {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": polygon_coordinates,
                    }
                },
            }
        }
        if date_from:
            payload["input"]["date_from"] = date_from
            payload["input"]["date_to"] = date_to
            payload["input"]["date_range_type"] = date_range_type or "any"
        if extensions:
            payload["extensions"] = extensions
        if filters:
            payload["filters"] = filters
        return await paginate_search(self._post, payload)

    async def applications(
        self,
        date_from: str,
        date_to: str,
        council_ids: list[int] | None = None,
        *,
        date_range_type: str = "any",
        extensions: dict | None = None,
        filters: dict | None = None,
    ) -> list[ApplicationsResponse]:
        """Fetch planning applications by date range across all councils or a subset.

        Args:
            date_from: ISO date string, e.g. "2025-01-01".
            date_to: ISO date string, e.g. "2025-03-31".
            council_ids: Optional list of council IDs to restrict the query.
            date_range_type: "any", "decided", or "validated". Defaults to "any".
            extensions: Optional dict of extension flags.
            filters: Optional dict of field filters.

        Returns:
            List of ApplicationsResponse objects from all pages.

        Raises:
            IbexAuthError: API returned 403.
            PayloadTooLargeError: API returned 413 (no auto-subdivision).
            httpx.HTTPStatusError: Unrecoverable 5xx after all retries.
        """
        payload: dict = {
            "input": {
                "date_from": date_from,
                "date_to": date_to,
                "date_range_type": date_range_type,
            }
        }
        if council_ids:
            payload["input"]["council_id"] = council_ids
        if extensions:
            payload["extensions"] = extensions
        if filters:
            payload["filters"] = filters
        return await paginate_applications(self._post, payload)

    async def stats(
        self,
        council_id: int,
        date_from: str,
        date_to: str,
    ) -> StatsResponse:
        """Retrieve planning statistics for a single council and date range.

        Args:
            council_id: Numeric council identifier.
            date_from: ISO date string, e.g. "2025-01-01".
            date_to: ISO date string, e.g. "2025-12-31".

        Returns:
            StatsResponse with approval rate, decision times, etc.

        Raises:
            IbexAuthError: API returned 403.
            PayloadTooLargeError: API returned 413 (no auto-subdivision).
            httpx.HTTPStatusError: Unrecoverable 5xx after all retries.
        """
        payload = {
            "input": {
                "council_id": council_id,
                "date_from": date_from,
                "date_to": date_to,
            }
        }
        response = await self._post("/stats", payload)
        return StatsResponse.model_validate(response.json())
