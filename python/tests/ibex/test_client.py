"""Unit tests for IbexClient using respx HTTP mocks.

All tests use the mock_ibex fixture from conftest.py which intercepts
httpx.AsyncClient calls at the transport level via respx.mock.

The TEST_SETTINGS instance uses a placeholder API key; no live network calls
are made.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from hashbrowns.config import Settings
from hashbrowns.ibex.client import IbexAuthError, IbexClient, SearchAreaTooSmallError
from hashbrowns.ibex.models import ApplicationsResponse, SearchResponse, StatsResponse

BASE_URL = "https://ibex.seractech.co.uk"

TEST_SETTINGS = Settings(
    ibex_api_key="test-jwt",
    ibex_base_url=BASE_URL,
    ibex_max_concurrency=2,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_client() -> IbexClient:
    """Return an IbexClient configured against the test base URL."""
    return IbexClient(TEST_SETTINGS)


# ---------------------------------------------------------------------------
# Test 1: search returns list[SearchResponse]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_search_responses(
    mock_ibex, search_response_one_result
):
    """search() returns a list containing one SearchResponse on a 200 reply."""
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )

    async with _make_client() as client:
        results = await client.search([528349, 186246], 300, 27700)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], SearchResponse)
    assert results[0].council_name == "Camden"


# ---------------------------------------------------------------------------
# Test 2: search_polygon returns results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_polygon_returns_results(
    mock_ibex, search_response_one_result
):
    """search_polygon() returns results for a polygon payload."""
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )
    polygon = [
        [
            [528000, 186000],
            [529000, 186000],
            [529000, 187000],
            [528000, 186000],
        ]
    ]

    async with _make_client() as client:
        results = await client.search_polygon(polygon, 27700)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert isinstance(results[0], SearchResponse)


# ---------------------------------------------------------------------------
# Test 3: applications returns list[ApplicationsResponse]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_applications_returns_applications_responses(
    mock_ibex, applications_response_fixture
):
    """applications() returns one ApplicationsResponse on a 200 reply."""
    mock_ibex.post("/applications").mock(
        return_value=httpx.Response(200, json=applications_response_fixture)
    )

    async with _make_client() as client:
        results = await client.applications("2025-01-01", "2025-03-31")

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], ApplicationsResponse)
    assert results[0].council_name == "Rochdale"


# ---------------------------------------------------------------------------
# Test 4: stats returns StatsResponse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_returns_stats_response(mock_ibex, stats_response_fixture):
    """stats() returns a StatsResponse with approval_rate == 85.5."""
    mock_ibex.post("/stats").mock(
        return_value=httpx.Response(200, json=stats_response_fixture)
    )

    async with _make_client() as client:
        result = await client.stats(240, "2025-01-01", "2025-12-31")

    assert isinstance(result, StatsResponse)
    assert result.approval_rate == 85.5


# ---------------------------------------------------------------------------
# Test 5: 403 raises IbexAuthError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_403_raises_auth_error(mock_ibex):
    """A 403 response raises IbexAuthError without tenacity retrying."""
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(403, json={"message": "Forbidden"})
    )

    async with _make_client() as client:
        with pytest.raises(IbexAuthError):
            await client.search([528349, 186246], 300, 27700)


# ---------------------------------------------------------------------------
# Test 6: 413 triggers radius subdivision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_413_triggers_radius_subdivision(
    mock_ibex, search_response_one_result
):
    """On first 413, search() halves the radius and retries; result returned."""
    call_count = 0

    def search_side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(413, json={"message": "too large", "error": "too large"})
        return httpx.Response(200, json=search_response_one_result)

    mock_ibex.post("/search").mock(side_effect=search_side_effect)

    async with _make_client() as client:
        results = await client.search([528349, 186246], 300, 27700)

    assert isinstance(results, list)
    assert len(results) == 1
    assert call_count == 2  # first attempt (413) + retry with halved radius


# ---------------------------------------------------------------------------
# Test 7: radius below minimum raises immediately (no request sent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_413_radius_below_minimum_raises(mock_ibex):
    """search() with radius < MIN_RADIUS_METRES raises SearchAreaTooSmallError immediately."""
    # Route registered but should NOT be called
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=[])
    )

    async with _make_client() as client:
        with pytest.raises(SearchAreaTooSmallError, match="below minimum"):
            await client.search([528349, 186246], 40, 27700)


# ---------------------------------------------------------------------------
# Test 8: too many subdivisions raises SearchAreaTooSmallError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_413_too_many_subdivisions_raises(mock_ibex):
    """After MAX_413_DEPTH halvings, SearchAreaTooSmallError is raised."""
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(413, json={"message": "too large"})
    )

    async with _make_client() as client:
        with pytest.raises(SearchAreaTooSmallError):
            # 2000 → 1000 → 500 → 250 → 125 → 62.5 → 31.25 (< 50, raises before request)
            # depth 0 → 1 → 2 → 3 → 4 → 5 (> MAX_413_DEPTH=4, raises)
            await client.search([528349, 186246], 2000, 27700)


# ---------------------------------------------------------------------------
# Test 9: filters serialised into request body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_filters(mock_ibex, search_response_one_result):
    """filters kwarg appears as 'filters' key in the POST request body."""
    route = mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )

    async with _make_client() as client:
        await client.search(
            [528349, 186246],
            300,
            27700,
            filters={"normalised_decision": ["Approved"]},
        )

    assert route.called
    import json
    sent_body = json.loads(route.calls[0].request.content)
    assert "filters" in sent_body
    assert sent_body["filters"] == {"normalised_decision": ["Approved"]}


# ---------------------------------------------------------------------------
# Test 10: extensions serialised into request body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_extensions(mock_ibex, search_response_one_result):
    """extensions kwarg appears as 'extensions' key in the POST request body."""
    route = mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )

    async with _make_client() as client:
        await client.search(
            [528349, 186246],
            300,
            27700,
            extensions={"appeals": True, "project_type": True},
        )

    assert route.called
    import json
    sent_body = json.loads(route.calls[0].request.content)
    assert "extensions" in sent_body
    assert sent_body["extensions"] == {"appeals": True, "project_type": True}


# ---------------------------------------------------------------------------
# Test 11: applications with council_ids serialised correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_applications_with_council_ids(
    mock_ibex, applications_response_fixture
):
    """council_ids are serialised as 'council_id' inside the input dict."""
    route = mock_ibex.post("/applications").mock(
        return_value=httpx.Response(200, json=applications_response_fixture)
    )

    async with _make_client() as client:
        await client.applications("2025-01-01", "2025-03-31", council_ids=[240, 10])

    assert route.called
    import json
    sent_body = json.loads(route.calls[0].request.content)
    assert "council_id" in sent_body["input"]
    assert sent_body["input"]["council_id"] == [240, 10]


# ---------------------------------------------------------------------------
# Test 12: date_range_type forwarded in search payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_date_range_type_required_when_dates_provided(
    mock_ibex, search_response_one_result
):
    """date_range_type is serialised into the request body when dates provided."""
    route = mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )

    async with _make_client() as client:
        await client.search(
            [528349, 186246],
            300,
            27700,
            date_from="2025-01-01",
            date_to="2025-03-31",
            date_range_type="decided",
        )

    assert route.called
    import json
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["input"]["date_range_type"] == "decided"
    assert sent_body["input"]["date_from"] == "2025-01-01"
    assert sent_body["input"]["date_to"] == "2025-03-31"


# ---------------------------------------------------------------------------
# Test 13: context manager closes the underlying httpx client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_client(mock_ibex, search_response_one_result):
    """After exiting the async with block, client._client.is_closed is True."""
    mock_ibex.post("/search").mock(
        return_value=httpx.Response(200, json=search_response_one_result)
    )

    client = _make_client()
    async with client:
        # client is open inside the block
        assert not client._client.is_closed
        await client.search([528349, 186246], 300, 27700)

    # After exiting the context, httpx client must be closed
    assert client._client.is_closed
