"""Async HTTP client for the Planning API.

Provides typed methods for each API endpoint, returning validated Pydantic
models.  Uses ``httpx.AsyncClient`` with connection pooling, exponential-
backoff retry on transient errors, and an ``asyncio.Semaphore``-based
concurrency limiter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from config.settings import Settings, get_settings
from data.schema import (
    CouncilStats,
    LookupExtensions,
    LookupRequest,
    LookupResponse,
    PlanningApplication,
    SearchExtensions,
    SearchFilters,
    SearchInput,
    SearchRequest,
    SearchResponse,
    StatsInput,
    StatsRequest,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0


class PlanningAPIClient:
    """Async wrapper around the planning data REST API.

    Features:
    - **Connection pooling** via ``httpx.Limits(max_connections=20)``.
    - **Rate limiting** via ``asyncio.Semaphore`` (configurable, default 10).
    - **Exponential backoff** on 429 / 5xx responses (3 retries, base 1 s).
    - All public methods return validated Pydantic models.

    Use as an async context manager for automatic cleanup::

        async with PlanningAPIClient() as client:
            apps = await client.search_applications("council-01")

    Parameters:
        settings: Application settings (injected for testability).
        max_retries: Maximum retry attempts for transient errors.
        base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        max_retries: int = _MAX_RETRIES,
        base_delay: float = _BASE_DELAY_S,
    ) -> None:
        self._settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(
            self._settings.max_concurrent_requests,
        )
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._request_count: int = 0
        self._error_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (and lazily create) the underlying ``httpx.AsyncClient``.

        The client is configured with the base URL, auth header, connection
        pool limits, and a 30-second timeout.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._settings.planning_api_base_url,
                headers={
                    "Authorization": f"Bearer {self._settings.planning_api_auth_token}",
                    "Content-Type": "application/json",
                },
                limits=httpx.Limits(max_connections=20),
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client and log final stats."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            logger.info(
                "Client closed — %d requests sent, %d errors encountered",
                self._request_count,
                self._error_count,
            )

    async def __aenter__(self) -> PlanningAPIClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Internal request helper ───────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
    ) -> httpx.Response:
        """Send an HTTP request with semaphore rate-limiting and retry.

        Retries up to ``_max_retries`` times with exponential backoff when
        the server responds with 429 (Too Many Requests) or any 5xx status,
        or when a transport-level error (timeout, connection reset) occurs.

        Args:
            method: HTTP method (``GET``, ``POST``, etc.).
            path: URL path relative to the base URL.
            json: Optional JSON body.

        Returns:
            The successful ``httpx.Response``.

        Raises:
            httpx.HTTPStatusError: On a non-retryable 4xx or after exhausting
                retries on a retryable status.
            httpx.TransportError: After exhausting retries on a transport
                error.
        """
        client = await self._get_client()

        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                self._request_count += 1
                logger.debug(
                    "Request #%d  %s %s  (attempt %d/%d)",
                    self._request_count,
                    method,
                    path,
                    attempt + 1,
                    self._max_retries + 1,
                )

                # --- transport-level errors (timeout, refused, etc.) ---
                try:
                    response = await client.request(method, path, json=json)
                except httpx.TransportError as exc:
                    self._error_count += 1
                    if attempt < self._max_retries:
                        delay = self._base_delay * (2**attempt)
                        logger.warning(
                            "Transport error on %s %s — %s — retrying in %.1fs "
                            "(attempt %d/%d)",
                            method,
                            path,
                            exc,
                            delay,
                            attempt + 1,
                            self._max_retries + 1,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

                # --- retryable HTTP status codes ---
                if response.status_code == 429 or response.status_code >= 500:
                    self._error_count += 1
                    if attempt < self._max_retries:
                        delay = self._base_delay * (2**attempt)
                        logger.warning(
                            "Retryable HTTP %d on %s %s — retrying in %.1fs "
                            "(attempt %d/%d)",
                            response.status_code,
                            method,
                            path,
                            delay,
                            attempt + 1,
                            self._max_retries + 1,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # exhausted retries — raise status error
                    response.raise_for_status()

                # --- non-retryable 4xx ---
                response.raise_for_status()
                return response

        # Unreachable — the loop always returns or raises — but keeps mypy
        # happy when the semaphore path confuses control-flow analysis.
        raise RuntimeError("Retry loop exited without returning or raising")

    # ── Endpoint 1: search applications ───────────────────────────────

    async def _search_raw(self, request: SearchRequest) -> SearchResponse:
        """Execute a search and return the full response envelope."""
        response = await self._request(
            "POST",
            "/search",
            json=request.model_dump(by_alias=True, exclude_none=True),
        )
        return SearchResponse.model_validate(response.json())

    async def search_applications(
        self,
        council_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date_range_type: str = "determined",
        extensions: Optional[SearchExtensions] = None,
        filters: Optional[SearchFilters] = None,
    ) -> list[PlanningApplication]:
        """Search planning applications for a given council (single page).

        Args:
            council_id: Local planning authority identifier.
            page: Page number (1-based).
            page_size: Number of results per page.
            date_from: ISO-8601 start date filter (inclusive).
            date_to: ISO-8601 end date filter (inclusive).
            date_range_type: Which date field the range applies to
                (default ``"determined"``).
            extensions: Optional extension toggles (documents, appeals, etc.).
            filters: Optional filters (application type, decision, keywords).

        Returns:
            List of validated ``PlanningApplication`` models for the
            requested page.
        """
        request = SearchRequest(
            input=SearchInput(
                council_id=council_id,
                page=page,
                page_size=page_size,
                date_from=date_from,
                date_to=date_to,
                date_range_type=date_range_type,
            ),
            extensions=extensions or SearchExtensions(),
            filters=filters or SearchFilters(),
        )
        result = await self._search_raw(request)
        return result.applications

    async def search_all_pages(
        self,
        council_id: str,
        *,
        page_size: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date_range_type: str = "determined",
        extensions: Optional[SearchExtensions] = None,
        filters: Optional[SearchFilters] = None,
    ) -> list[PlanningApplication]:
        """Auto-paginate through *all* search results for a query.

        Fetches page 1 to discover ``total_results``, then fetches every
        remaining page concurrently (bounded by the semaphore).

        Args:
            council_id: Local planning authority identifier.
            page_size: Results per page.
            date_from: ISO-8601 start date filter (inclusive).
            date_to: ISO-8601 end date filter (inclusive).
            date_range_type: Which date field the range applies to.
            extensions: Optional extension toggles.
            filters: Optional filters.

        Returns:
            Complete list of ``PlanningApplication`` models across all pages.
        """
        ext = extensions or SearchExtensions()
        flt = filters or SearchFilters()

        def _build_request(page: int) -> SearchRequest:
            return SearchRequest(
                input=SearchInput(
                    council_id=council_id,
                    page=page,
                    page_size=page_size,
                    date_from=date_from,
                    date_to=date_to,
                    date_range_type=date_range_type,
                ),
                extensions=ext,
                filters=flt,
            )

        # Page 1 — discover total
        first = await self._search_raw(_build_request(1))
        all_applications: list[PlanningApplication] = list(first.applications)
        total = first.total_results
        total_pages = (total + page_size - 1) // page_size

        logger.info(
            "search_all_pages: %d total results across %d pages",
            total,
            total_pages,
        )

        if total_pages <= 1:
            return all_applications

        # Remaining pages — fetch concurrently
        async def _fetch_page(page: int) -> list[PlanningApplication]:
            resp = await self._search_raw(_build_request(page))
            return resp.applications

        remaining = await asyncio.gather(
            *(_fetch_page(p) for p in range(2, total_pages + 1)),
        )
        for page_apps in remaining:
            all_applications.extend(page_apps)

        logger.info(
            "search_all_pages: fetched %d applications total",
            len(all_applications),
        )
        return all_applications

    # ── Endpoint 2: application lookup by reference ───────────────────

    async def lookup_applications(
        self,
        applications: list[tuple[str, str]],
        *,
        extensions: Optional[LookupExtensions] = None,
    ) -> list[PlanningApplication]:
        """Look up applications by ``(council_id, planning_reference)`` pairs.

        When ``extensions.documents`` is ``True`` the returned applications
        will include ``DocumentMetadata`` entries with full S3 download links.

        Args:
            applications: Pairs of ``(council_id, planning_reference)``.
            extensions: Optional extension toggles.

        Returns:
            List of validated ``PlanningApplication`` models with populated
            document links.
        """
        request = LookupRequest(
            applications=[list(pair) for pair in applications],
            extensions=extensions or LookupExtensions(),
        )
        response = await self._request(
            "POST",
            "/applications",
            json=request.model_dump(by_alias=True, exclude_none=True),
        )
        result = LookupResponse.model_validate(response.json())
        return result.applications

    # ── Endpoint 3: council stats ─────────────────────────────────────

    async def get_council_stats(
        self,
        council_id: str,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> CouncilStats:
        """Fetch aggregated statistics for a council.

        Returns approval/refusal rates, average decision times by project
        type, application counts, new-homes approved, and an overall
        development activity level.

        Args:
            council_id: Local planning authority identifier.
            date_from: ISO-8601 start date for the stats window.
            date_to: ISO-8601 end date for the stats window.

        Returns:
            A validated ``CouncilStats`` model.
        """
        request = StatsRequest(
            input=StatsInput(
                council_id=council_id,
                date_from=date_from,
                date_to=date_to,
            ),
        )
        response = await self._request(
            "POST",
            "/stats",
            json=request.model_dump(by_alias=True, exclude_none=True),
        )
        return CouncilStats.model_validate(response.json())
