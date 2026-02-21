"""TDD tests for pagination behaviour and 413 handling in ibex pagination helpers."""
from __future__ import annotations

import copy
import json

import httpx
import pytest

from hashbrowns.ibex.pagination import (
    PayloadTooLargeError,
    paginate_applications,
    paginate_search,
)


# ---------------------------------------------------------------------------
# Helpers: minimal valid fixture dicts
# ---------------------------------------------------------------------------

SEARCH_ITEM = {
    "council_id": 240,
    "council_name": "Camden",
    "planning_reference": "2025/0970/P",
    "url": "https://planningrecords.camden.gov.uk/example",
    "normalised_decision": "Approved",
    "geometry": "POINT(528349 186246)",
}

APPLICATIONS_ITEM = {
    "council_id": 10,
    "council_name": "Rochdale",
    "planning_reference": "24/00057/FUL",
    "url": "https://publicaccess.rochdale.gov.uk/example",
    "normalised_decision": "Approved",
    "geometry": "POLYGON((389887.2 413707.05,389882.9 413704.2,389887.2 413707.05))",
}


def _search_page(n: int) -> list[dict]:
    """Return a page of n minimal search result dicts."""
    return [copy.deepcopy(SEARCH_ITEM) for _ in range(n)]


def _applications_page(n: int) -> list[dict]:
    """Return a page of n minimal application result dicts."""
    return [copy.deepcopy(APPLICATIONS_ITEM) for _ in range(n)]


def _json_response(data: object, status_code: int = 200) -> httpx.Response:
    """Build a real httpx.Response with JSON body and a dummy request attached.

    httpx.Response.raise_for_status() requires a request to be set on the
    response; without it a RuntimeError is raised even for 2xx responses.
    """
    content = json.dumps(data).encode()
    response = httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json"},
    )
    # Attach a dummy request so raise_for_status() works correctly.
    response.request = httpx.Request("POST", "https://ibex.seractech.co.uk/search")
    return response


def _make_post_fn(responses: list[httpx.Response]):
    """Return an async post_fn that yields responses from the list in order."""
    call_count = 0

    async def post_fn(path: str, payload: dict) -> httpx.Response:
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    return post_fn


# ---------------------------------------------------------------------------
# paginate_search tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_page_search():
    """Single page with 3 results (< default page_size=5000) returns 3 SearchResponse."""
    page_data = _search_page(3)
    post_fn = _make_post_fn([_json_response(page_data)])

    results = await paginate_search(post_fn, {"input": {}})

    assert len(results) == 3


@pytest.mark.asyncio
async def test_multi_page_search():
    """Full page 1 (page_size=3) then 2 results on page 2 → 5 total SearchResponse."""
    page_size = 3
    page1 = _search_page(page_size)
    page2 = _search_page(2)
    post_fn = _make_post_fn([_json_response(page1), _json_response(page2)])

    results = await paginate_search(post_fn, copy.deepcopy({"input": {}}), page_size=page_size)

    assert len(results) == page_size + 2


@pytest.mark.asyncio
async def test_empty_search_result():
    """Empty first page → returns empty list."""
    post_fn = _make_post_fn([_json_response([])])

    results = await paginate_search(post_fn, {"input": {}})

    assert results == []


@pytest.mark.asyncio
async def test_413_raises_payload_too_large():
    """413 response on first call raises PayloadTooLargeError immediately."""
    response_413 = _json_response({"message": "too large", "error": "too large"}, status_code=413)
    post_fn = _make_post_fn([response_413])

    with pytest.raises(PayloadTooLargeError) as exc_info:
        await paginate_search(post_fn, {"input": {}})

    assert exc_info.value.response.status_code == 413


# ---------------------------------------------------------------------------
# paginate_applications tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_page_applications():
    """2 application results on single page returns 2 ApplicationsResponse."""
    page_data = _applications_page(2)
    post_fn = _make_post_fn([_json_response(page_data)])

    results = await paginate_applications(post_fn, {"input": {}})

    assert len(results) == 2


@pytest.mark.asyncio
async def test_multi_page_applications():
    """Full page 1 (1000 items) + 5 items on page 2 → 1005 ApplicationsResponse."""
    page_size = 1000
    page1 = _applications_page(page_size)
    page2 = _applications_page(5)
    post_fn = _make_post_fn([_json_response(page1), _json_response(page2)])

    results = await paginate_applications(post_fn, copy.deepcopy({"input": {}}))

    assert len(results) == 1005


@pytest.mark.asyncio
async def test_413_applications_raises():
    """413 on applications raises PayloadTooLargeError."""
    response_413 = _json_response({"message": "too large", "error": "too large"}, status_code=413)
    post_fn = _make_post_fn([response_413])

    with pytest.raises(PayloadTooLargeError):
        await paginate_applications(post_fn, {"input": {}})


# ---------------------------------------------------------------------------
# Payload inspection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_numbers_sent():
    """Verify page=1 is sent on the first call, page=2 on the second call."""
    page_size = 3
    page1 = _search_page(page_size)
    page2 = _search_page(1)

    calls: list[dict] = []

    async def capturing_post_fn(path: str, payload: dict) -> httpx.Response:
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            return _json_response(page1)
        return _json_response(page2)

    await paginate_search(capturing_post_fn, copy.deepcopy({"input": {}}), page_size=page_size)

    assert calls[0]["input"]["page"] == 1
    assert calls[1]["input"]["page"] == 2


@pytest.mark.asyncio
async def test_page_size_in_payload():
    """Verify page_size=5000 (default) appears in search payload input."""
    calls: list[dict] = []

    async def capturing_post_fn(path: str, payload: dict) -> httpx.Response:
        calls.append(copy.deepcopy(payload))
        return _json_response(_search_page(1))

    await paginate_search(capturing_post_fn, {"input": {}})

    assert calls[0]["input"]["page_size"] == 5000
