"""Async HTTP client for planning data, backed by the working IbexClient.

Wraps :class:`hashbrowns.ibex.client.IbexClient` from the sibling ``python``
package and adapts its response models to the planning-oracle schema types so
that all consuming code (training, dataset, pipeline) continues to work
unchanged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from config.settings import Settings, get_settings
from data.schema import (
    CouncilStats,
    PlanningApplication,
)

# ---------------------------------------------------------------------------
# Import IbexClient from the sibling python package
# ---------------------------------------------------------------------------

_PYTHON_PKG_DIR = str(Path(__file__).resolve().parents[2] / "python")
if _PYTHON_PKG_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_PKG_DIR)

from hashbrowns.ibex.client import IbexClient  # noqa: E402
from hashbrowns.ibex.models import (  # noqa: E402
    ApplicationsResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response converters
# ---------------------------------------------------------------------------


def _ibex_app_to_schema(app: ApplicationsResponse) -> PlanningApplication:
    """Convert an Ibex ``ApplicationsResponse`` to planning-oracle's
    ``PlanningApplication`` schema model."""

    # Map proposed_unit_mix fields
    proposed_units = None
    if app.proposed_unit_mix is not None:
        from data.schema import ProposedUnitMix

        mix = app.proposed_unit_mix
        proposed_units = ProposedUnitMix(
            one_bed=mix.proposed_1_bed_units or 0,
            two_bed=mix.proposed_2_bed_units or 0,
            three_bed=mix.proposed_3_bed_units or 0,
            four_plus_bed=mix.proposed_4_plus_bed_units or 0,
            affordable=mix.affordable_housing_units or 0,
        )

    # Map proposed_floor_area fields
    proposed_floor_area = None
    if app.proposed_floor_area is not None:
        from data.schema import ProposedFloorArea

        fa = app.proposed_floor_area
        proposed_floor_area = ProposedFloorArea(
            gross_internal_area_to_add_sqm=fa.gross_internal_area_to_add_sqm or 0.0,
            proposed_gross_floor_area_sqm=fa.proposed_gross_floor_area_sqm or 0.0,
        )

    return PlanningApplication(
        council_id=app.council_id,
        council_name=app.council_name,
        planning_reference=app.planning_reference,
        proposal=app.proposal,
        raw_address=app.raw_address,
        raw_application_type=app.raw_application_type,
        normalised_application_type=(
            app.normalised_application_type.value
            if app.normalised_application_type
            else None
        ),
        project_type=(
            app.project_type.value if app.project_type else None
        ),
        heading=app.heading,
        normalised_decision=(
            app.normalised_decision.value
            if app.normalised_decision
            else None
        ),
        raw_decision=app.raw_decision,
        application_date=app.application_date,
        decided_date=app.decided_date,
        proposed_unit_mix=proposed_units,
        proposed_floor_area=proposed_floor_area,
        num_new_houses=app.num_new_houses,
    )


def _ibex_stats_to_schema(
    stats: StatsResponse, council_id: str,
) -> CouncilStats:
    """Convert an Ibex ``StatsResponse`` to planning-oracle's
    ``CouncilStats`` schema model."""

    # Convert typed AverageDecisionTime to a plain dict
    avg_time = stats.average_decision_time
    avg_time_dict = {
        k: v
        for k, v in {
            "small residential": avg_time.small_residential,
            "tree": avg_time.tree,
            "large residential": avg_time.large_residential,
            "home improvement": avg_time.home_improvement,
            "mixed": avg_time.mixed,
            "medium residential": avg_time.medium_residential,
        }.items()
        if v is not None
    }

    # Convert typed NumberOfApplications to a plain dict
    num_apps = stats.number_of_applications
    num_apps_dict = {
        k: v
        for k, v in {
            "non-material amendment": num_apps.non_material_amendment,
            "discharge of conditions": num_apps.discharge_of_conditions,
            "listed building consent": num_apps.listed_building_consent,
            "advertisement consent": num_apps.advertisement_consent,
            "householder planning application": num_apps.householder_planning_application,
            "tree preservation order": num_apps.tree_preservation_order,
            "lawful development": num_apps.lawful_development,
            "change of use": num_apps.change_of_use,
            "full planning application": num_apps.full_planning_application,
            "conservation area": num_apps.conservation_area,
            "utilities": num_apps.utilities,
            "unknown": num_apps.unknown,
            "environmental impact": num_apps.environmental_impact,
            "section 106": num_apps.section_106,
            "pre-application": num_apps.pre_application,
            "other": num_apps.other,
        }.items()
        if v is not None
    }

    return CouncilStats(
        council_id=council_id,
        approval_rate=stats.approval_rate,
        refusal_rate=stats.refusal_rate,
        average_decision_time=avg_time_dict,
        number_of_applications=num_apps_dict,
        number_of_new_homes_approved=stats.number_of_new_homes_approved,
        council_development_activity_level=(
            stats.council_development_activity_level.value
        ),
    )


# ---------------------------------------------------------------------------
# Ibex Settings adapter
# ---------------------------------------------------------------------------


class _IbexSettings:
    """Minimal settings object compatible with the IbexClient constructor."""

    def __init__(self, settings: Settings) -> None:
        self.ibex_api_key = settings.ibex_api_key
        self.ibex_base_url = settings.ibex_base_url
        self.ibex_max_concurrency = settings.ibex_max_concurrency


# ---------------------------------------------------------------------------
# PlanningAPIClient — public interface backed by IbexClient
# ---------------------------------------------------------------------------


class PlanningAPIClient:
    """Async wrapper around the Ibex planning data API.

    Drop-in replacement for the old PlanningAPIClient. Uses the working
    :class:`IbexClient` from the ``python`` package under the hood.

    Use as an async context manager::

        async with PlanningAPIClient() as client:
            apps = await client.search_all_pages("council-01")
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ibex = IbexClient(_IbexSettings(self._settings))

    async def __aenter__(self) -> PlanningAPIClient:
        await self._ibex.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._ibex.__aexit__(*exc)

    # ── search_all_pages ──────────────────────────────────────────────

    async def search_all_pages(
        self,
        council_id: Union[int, str],
        *,
        page_size: int = 1000,
        max_pages: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date_range_type: str = "determined",
        extensions: object = None,
        filters: object = None,
    ) -> list[PlanningApplication]:
        """Fetch applications for a council via the Ibex /applications
        endpoint, converting results to planning-oracle schema models.

        Paginates up to *max_pages* pages of *page_size* results each
        (default: 10 pages x 1000 = 10,000 applications max).
        """

        council_id_str = str(council_id)

        ibex_extensions = None
        if extensions is not None:
            ibex_extensions = {
                "project_type": True,
                "heading": True,
            }

        ibex_filters = None
        if filters is not None:
            ibex_filters = {}

        d_from = date_from or "2015-01-01"
        d_to = date_to or "2030-12-31"

        payload: dict = {
            "input": {
                "date_from": d_from,
                "date_to": d_to,
                "date_range_type": date_range_type if date_range_type != "determined" else "any",
            }
        }
        if council_id_str.isdigit():
            payload["input"]["council_id"] = [int(council_id_str)]
        if ibex_extensions:
            payload["extensions"] = ibex_extensions
        if ibex_filters:
            payload["filters"] = ibex_filters

        from hashbrowns.ibex.models import ApplicationsResponse as IbexAppResponse

        all_responses: list[IbexAppResponse] = []
        for page in range(1, max_pages + 1):
            payload["input"]["page"] = page
            payload["input"]["page_size"] = page_size

            response = await self._ibex._post("/applications", payload)
            batch = [IbexAppResponse.model_validate(item) for item in response.json()]
            all_responses.extend(batch)

            logger.info(
                "  page %d/%d: %d results", page, max_pages, len(batch),
            )
            if len(batch) < page_size:
                break

        applications = [_ibex_app_to_schema(r) for r in all_responses]
        logger.info(
            "search_all_pages: fetched %d applications for council %s (%d pages)",
            len(applications),
            council_id,
            page,
        )
        return applications

    # ── search_applications (single page — kept for compatibility) ────

    async def search_applications(
        self,
        council_id: Union[int, str],
        *,
        page: int = 1,
        page_size: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date_range_type: str = "determined",
        extensions: object = None,
        filters: object = None,
    ) -> list[PlanningApplication]:
        """Search applications (delegates to search_all_pages)."""
        return await self.search_all_pages(
            council_id,
            page_size=page_size,
            date_from=date_from,
            date_to=date_to,
            date_range_type=date_range_type,
            extensions=extensions,
            filters=filters,
        )

    # ── get_council_stats ─────────────────────────────────────────────

    async def get_council_stats(
        self,
        council_id: Union[int, str],
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> CouncilStats:
        """Fetch council statistics via the Ibex /stats endpoint."""

        d_from = date_from or "2015-01-01"
        d_to = date_to or "2030-12-31"

        council_id_str = str(council_id)
        council_id_int = int(council_id_str) if council_id_str.isdigit() else 0

        stats = await self._ibex.stats(
            council_id=council_id_int,
            date_from=d_from,
            date_to=d_to,
        )

        result = _ibex_stats_to_schema(stats, council_id_str)
        logger.info("get_council_stats: fetched stats for council %s", council_id_str)
        return result

    # ── lookup_applications (kept for compatibility) ──────────────────

    async def lookup_applications(
        self,
        applications: list[tuple[str, str]],
        *,
        extensions: object = None,
    ) -> list[PlanningApplication]:
        """Look up applications by reference.

        Note: The Ibex API doesn't have a direct lookup endpoint, so this
        fetches all applications for each council and filters by reference.
        """
        results: list[PlanningApplication] = []
        # Group by council_id
        by_council: dict[str, list[str]] = {}
        for council_id, ref in applications:
            by_council.setdefault(council_id, []).append(ref)

        for council_id, refs in by_council.items():
            all_apps = await self.search_all_pages(council_id)
            ref_set = set(refs)
            results.extend(
                app for app in all_apps
                if app.planning_reference in ref_set
            )

        return results
