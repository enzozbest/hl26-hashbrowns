"""Tests for the async Planning API client."""

from __future__ import annotations

import pytest

from data.api_client import PlanningAPIClient
from data.schema import PlanningApplication


class TestPlanningAPIClient:
    """Test suite for ``PlanningAPIClient``."""

    @pytest.mark.asyncio
    async def test_search_applications_returns_list(
        self, test_settings
    ) -> None:
        """search_applications should return a list of PlanningApplication."""
        # TODO: Mock httpx responses and verify deserialization
        ...

    @pytest.mark.asyncio
    async def test_get_council_stats_returns_model(
        self, test_settings
    ) -> None:
        """get_council_stats should return a validated CouncilStats model."""
        # TODO: Mock httpx responses and verify deserialization
        ...

    @pytest.mark.asyncio
    async def test_get_document_returns_content(
        self, test_settings
    ) -> None:
        """get_document should return an ApplicationDocument with content."""
        # TODO: Mock httpx responses and verify content extraction
        ...

    @pytest.mark.asyncio
    async def test_client_sends_auth_header(self, test_settings) -> None:
        """The client should include the auth token in request headers."""
        # TODO: Inspect request headers in mock
        ...
