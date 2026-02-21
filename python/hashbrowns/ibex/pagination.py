"""Pagination helpers for Ibex API /search and /applications endpoints."""
from __future__ import annotations

from typing import Callable, Awaitable

import httpx

from hashbrowns.ibex.models import SearchResponse, ApplicationsResponse

MAX_PAGES = 1000


class PayloadTooLargeError(Exception):
    """Raised when the API returns HTTP 413 — payload is too large."""

    def __init__(self, response: httpx.Response):
        self.response = response
        super().__init__(f"413 Payload Too Large: {response.text[:200]}")


PostFn = Callable[[str, dict], Awaitable[httpx.Response]]


async def paginate_search(
    post_fn: PostFn,
    payload: dict,
    page_size: int = 5000,
) -> list[SearchResponse]:
    """Fetch all pages from /search, assembling SearchResponse objects.

    Terminates when a page returns fewer results than page_size (last page
    detected). A circuit breaker raises RuntimeError after MAX_PAGES pages to
    prevent infinite loops.

    Args:
        post_fn: Async callable (path: str, payload: dict) -> httpx.Response.
        payload: Request payload dict. Will be mutated (page/page_size added to
            payload["input"]). Pass a copy if the original must be preserved.
        page_size: Number of results per page. Defaults to 5000.

    Returns:
        Combined list of SearchResponse objects across all pages.

    Raises:
        PayloadTooLargeError: If any response has status 413.
        RuntimeError: If pagination exceeds MAX_PAGES pages.
        httpx.HTTPStatusError: For other non-2xx responses.
    """
    results: list[SearchResponse] = []
    page = 1

    while True:
        payload.setdefault("input", {})
        payload["input"]["page"] = page
        payload["input"]["page_size"] = page_size

        response = await post_fn("/search", payload)

        if response.status_code == 413:
            raise PayloadTooLargeError(response)
        response.raise_for_status()

        batch = [SearchResponse.model_validate(item) for item in response.json()]
        results.extend(batch)

        if len(batch) < page_size:
            break

        page += 1
        if page > MAX_PAGES:
            raise RuntimeError(
                f"Pagination exceeded {MAX_PAGES} pages — possible infinite loop. "
                "Check API response or reduce page_size."
            )

    return results


async def paginate_applications(
    post_fn: PostFn,
    payload: dict,
    page_size: int = 1000,
) -> list[ApplicationsResponse]:
    """Fetch all pages from /applications, assembling ApplicationsResponse objects.

    Terminates when a page returns fewer results than page_size (last page
    detected). A circuit breaker raises RuntimeError after MAX_PAGES pages to
    prevent infinite loops.

    Args:
        post_fn: Async callable (path: str, payload: dict) -> httpx.Response.
        payload: Request payload dict. Will be mutated (page/page_size added to
            payload["input"]). Pass a copy if the original must be preserved.
        page_size: Number of results per page. Defaults to 1000.

    Returns:
        Combined list of ApplicationsResponse objects across all pages.

    Raises:
        PayloadTooLargeError: If any response has status 413.
        RuntimeError: If pagination exceeds MAX_PAGES pages.
        httpx.HTTPStatusError: For other non-2xx responses.
    """
    results: list[ApplicationsResponse] = []
    page = 1

    while True:
        payload.setdefault("input", {})
        payload["input"]["page"] = page
        payload["input"]["page_size"] = page_size

        response = await post_fn("/applications", payload)

        if response.status_code == 413:
            raise PayloadTooLargeError(response)
        response.raise_for_status()

        batch = [ApplicationsResponse.model_validate(item) for item in response.json()]
        results.extend(batch)

        if len(batch) < page_size:
            break

        page += 1
        if page > MAX_PAGES:
            raise RuntimeError(
                f"Pagination exceeded {MAX_PAGES} pages — possible infinite loop. "
                "Check API response or reduce page_size."
            )

    return results
