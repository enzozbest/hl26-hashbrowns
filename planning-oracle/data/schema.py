"""Pydantic models matching the Planning API request and response shapes.

Each model mirrors the JSON structure of a specific API endpoint.  The API
uses **snake_case** keys on the wire, so field names match directly.  Aliases
are only needed where Python names differ from JSON keys.

All models set ``populate_by_name=True`` so instances can be constructed with
either the Python name or the JSON alias.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Nested / reusable models ────────────────────────────────────────────────


class ProposedUnitMix(BaseModel):
    """Breakdown of proposed residential units by bedroom count."""

    model_config = ConfigDict(populate_by_name=True)

    one_bed: int = Field(0, description="Number of 1-bedroom units")
    two_bed: int = Field(0, description="Number of 2-bedroom units")
    three_bed: int = Field(0, description="Number of 3-bedroom units")
    four_plus_bed: int = Field(0, description="Number of 4+ bedroom units")
    affordable: int = Field(0, description="Number of affordable housing units")


class ProposedFloorArea(BaseModel):
    """Proposed floor area details for the application.

    Matches the API response structure with five area measurements.
    """

    model_config = ConfigDict(populate_by_name=True)

    gross_internal_area_to_add_sqm: float = Field(
        0.0, description="Gross internal area to be added (sqm)"
    )
    existing_gross_floor_area_sqm: float = Field(
        0.0, description="Existing gross floor area (sqm)"
    )
    proposed_gross_floor_area_sqm: float = Field(
        0.0, description="Proposed gross floor area (sqm)"
    )
    floor_area_to_be_lost_sqm: float = Field(
        0.0, description="Floor area to be lost (sqm)"
    )
    floor_area_to_be_gained_sqm: float = Field(
        0.0, description="Floor area to be gained (sqm)"
    )


class SearchDocumentMetadata(BaseModel):
    """Document metadata as returned by the *search* endpoint.

    The search endpoint returns documents under the ``document_metadata``
    key with fields: date_published, document_type, description, document_link.
    """

    model_config = ConfigDict(populate_by_name=True)

    date_published: Optional[str] = Field(
        None, description="Date the document was published"
    )
    document_type: str = Field(
        "", description="Type/category of document (e.g. APPLICATION_FORM)"
    )
    description: str = Field("", description="Human-readable document description")
    document_link: str = Field("", description="URL to retrieve the document")


class LookupDocument(BaseModel):
    """Document as returned by the *lookup* (applications-by-id) endpoint.

    The lookup endpoint returns documents under the ``documents`` key with
    fields: id (int), s3Link (str), documentTypes (list[str]).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Unique document identifier")
    s3_link: str = Field("", alias="s3Link", description="S3 pre-signed URL")
    document_types: list[str] = Field(
        default_factory=list,
        alias="documentTypes",
        description="List of document type classifications",
    )


# ── Main data models ────────────────────────────────────────────────────────


class PlanningApplication(BaseModel):
    """Planning application record returned by the search and lookup endpoints.

    This is the primary unit of data in the pipeline.  Each record captures
    the proposal description, site details, decision outcome, and optional
    nested objects for unit mix, floor area, and attached documents.

    Fields are nullable because different endpoints populate different subsets.
    The API returns snake_case keys that map directly to field names.
    """

    model_config = ConfigDict(populate_by_name=True)

    # ── identifiers ────────────────────────────────────────────────────
    council_id: int = Field(
        ..., description="Local planning authority identifier"
    )
    council_name: Optional[str] = Field(
        None, description="Name of the local planning authority"
    )
    planning_reference: Optional[str] = Field(
        None, description="Human-readable planning reference (e.g. 2025/0970/P)"
    )
    url: Optional[str] = Field(
        None, description="URL to the application on the council's website"
    )

    # ── proposal / description ─────────────────────────────────────────
    proposal: Optional[str] = Field(
        None, description="Free-text proposal description"
    )
    heading: Optional[str] = Field(
        None, description="Short heading summarising the proposal"
    )

    # ── location ───────────────────────────────────────────────────────
    geometry: Optional[str] = Field(
        None, description="WKT geometry string (e.g. POINT(...))"
    )
    raw_address: Optional[str] = Field(
        None, description="Site address as recorded by the council"
    )
    centre_point: Optional[str] = Field(
        None, description="WKT centre point (from centre_point extension)"
    )

    # ── application types ──────────────────────────────────────────────
    raw_application_type: Optional[str] = Field(
        None, description="Raw application type from the council"
    )
    normalised_application_type: Optional[str] = Field(
        None, description="Normalised application type for filtering"
    )
    project_type: Optional[str] = Field(
        None, description="High-level project category (e.g. home improvement)"
    )

    # ── dates ──────────────────────────────────────────────────────────
    application_date: Optional[date] = Field(
        None, description="Date application was submitted/received"
    )
    decided_date: Optional[date] = Field(
        None, description="Date of decision"
    )

    # ── decision ───────────────────────────────────────────────────────
    raw_decision: Optional[str] = Field(
        None, description="Decision outcome as recorded by the council"
    )
    normalised_decision: Optional[str] = Field(
        None, description="Normalised decision value (Approved / Refused)"
    )

    # ── appeals ────────────────────────────────────────────────────────
    appeals: Optional[Any] = Field(
        None, description="Appeal information (null if no appeal)"
    )

    # ── housing numbers ────────────────────────────────────────────────
    num_new_houses: Optional[int] = Field(
        None, description="Number of net new houses proposed"
    )
    num_comments_received: Optional[int] = Field(
        None, description="Number of public comments received"
    )

    # ── nested objects ─────────────────────────────────────────────────
    proposed_unit_mix: Optional[ProposedUnitMix] = Field(
        None, description="Proposed residential unit mix"
    )
    proposed_floor_area: Optional[ProposedFloorArea] = Field(
        None, description="Proposed floor areas"
    )

    # ── documents (search endpoint uses document_metadata,
    #    lookup endpoint uses documents) ────────────────────────────────
    document_metadata: list[SearchDocumentMetadata] = Field(
        default_factory=list, description="Documents from the search endpoint"
    )
    documents: list[LookupDocument] = Field(
        default_factory=list, description="Documents from the lookup endpoint"
    )

    # ── convenience aliases for downstream code ────────────────────────

    @property
    def description(self) -> Optional[str]:
        """Alias for ``proposal`` used by feature extraction and embeddings."""
        return self.proposal

    @property
    def address(self) -> Optional[str]:
        """Alias for ``raw_address``."""
        return self.raw_address

    @property
    def application_type(self) -> Optional[str]:
        """Alias for ``raw_application_type``."""
        return self.raw_application_type

    @property
    def decision(self) -> Optional[str]:
        """Alias for ``raw_decision``."""
        return self.raw_decision

    @property
    def date_received(self) -> Optional[date]:
        """Alias for ``application_date`` used by temporal splits."""
        return self.application_date

    @property
    def decision_date(self) -> Optional[date]:
        """Alias for ``decided_date``."""
        return self.decided_date


class CouncilStats(BaseModel):
    """Aggregated statistics for a local planning authority.

    Returned by the stats endpoint.  Contains approval/refusal rates,
    decision speed broken down by project type, application counts, and
    an activity-level classification.

    Note: ``council_id`` and ``period_start``/``period_end`` are **not**
    returned by the API — they are injected by the client after the call.
    """

    model_config = ConfigDict(populate_by_name=True)

    council_id: Optional[int] = Field(
        None, description="Local planning authority identifier (injected by client)"
    )
    council_name: Optional[str] = Field(
        None, description="Name of the local planning authority"
    )
    approval_rate: Optional[float] = Field(
        None, description="Historical approval rate (0-100 percentage)"
    )
    refusal_rate: Optional[float] = Field(
        None, description="Historical refusal rate (0-100 percentage)"
    )
    average_decision_time: Optional[dict[str, float]] = Field(
        None,
        description="Average days to decision keyed by project type",
    )
    number_of_applications: Optional[dict[str, int]] = Field(
        None,
        description="Application counts keyed by normalised type",
    )
    number_of_new_homes_approved: Optional[int] = Field(
        None,
        description="Total net new homes approved in the period",
    )
    council_development_activity_level: Optional[str] = Field(
        None,
        description="Activity classification (high, medium, low)",
    )
    period_start: Optional[date] = Field(
        None, description="Stats period start date (injected by client)"
    )
    period_end: Optional[date] = Field(
        None, description="Stats period end date (injected by client)"
    )


# ── Request body models ─────────────────────────────────────────────────────

# Endpoint 1 — Search applications


class SearchInput(BaseModel):
    """``input`` block for the search endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    date_range_type: str = Field(
        "determined",
        description="Which date field to filter on (e.g. determined, validated)",
    )
    date_from: Optional[str] = Field(
        None, description="ISO-8601 start date"
    )
    date_to: Optional[str] = Field(
        None, description="ISO-8601 end date"
    )
    council_id: Optional[list[int]] = Field(
        None,
        description="Council identifier(s) — list of integer IDs",
    )
    coordinates: Optional[list[float]] = Field(
        None, description="[x, y] for radius search (units depend on srid)"
    )
    radius: Optional[float] = Field(
        None, description="Search radius (units determined by srid)"
    )
    srid: Optional[int] = Field(
        None, description="Spatial reference ID for the coordinates"
    )
    page: int = Field(1, description="Page number (1-based)")
    page_size: int = Field(100, description="Results per page")


class SearchExtensions(BaseModel):
    """``extensions`` block for the search endpoint — all boolean toggles."""

    model_config = ConfigDict(populate_by_name=True)

    appeals: bool = Field(False, description="Include appeal information")
    centre_point: bool = Field(False, description="Include centre point geometry")
    heading: bool = Field(False, description="Include proposal heading")
    unlimited_radius: bool = Field(
        False, description="Remove radius cap for spatial queries"
    )
    project_type: bool = Field(
        False, description="Include project type classification"
    )
    num_new_houses: bool = Field(
        False, description="Include number of new houses"
    )
    document_metadata: bool = Field(
        False, description="Include document metadata"
    )
    proposed_unit_mix: bool = Field(
        False, description="Include proposed residential unit mix"
    )
    proposed_floor_area: bool = Field(
        False, description="Include proposed floor area"
    )
    num_comments_received: bool = Field(
        False, description="Include number of public comments"
    )


class NumNewHousesFilter(BaseModel):
    """Min/max filter for number of new houses."""

    model_config = ConfigDict(populate_by_name=True)

    min: Optional[int] = Field(None, description="Minimum new houses (inclusive)")
    max: Optional[int] = Field(None, description="Maximum new houses (inclusive)")


class SearchFilters(BaseModel):
    """``filters`` block for the search endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    normalised_application_type: list[str] = Field(
        default_factory=list,
        description="Filter by normalised application types",
    )
    project_type: list[str] = Field(
        default_factory=list,
        description="Filter by project types",
    )
    normalised_decision: list[str] = Field(
        default_factory=list,
        description="Filter by normalised decisions",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Free-text keyword filters"
    )
    num_new_houses: Optional[NumNewHousesFilter] = Field(
        None,
        description="Min/max filter on proposed new houses",
    )


class SearchRequest(BaseModel):
    """Full request body for the search endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    input: SearchInput
    extensions: SearchExtensions = Field(default_factory=SearchExtensions)
    filters: SearchFilters = Field(default_factory=SearchFilters)


# Endpoint 2 — Application lookup by reference


class LookupExtensions(BaseModel):
    """``extensions`` block for the lookup (applications-by-id) endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    documents: bool = Field(False, description="Include documents with S3 links")
    appeals: bool = Field(False, description="Include appeal information")
    project_type: bool = Field(
        False, description="Include project type"
    )
    heading: bool = Field(False, description="Include proposal heading")


class LookupRequest(BaseModel):
    """Full request body for the application lookup endpoint.

    ``applications`` is a list of ``[council_id, planning_reference]`` pairs
    where council_id is an integer.
    """

    model_config = ConfigDict(populate_by_name=True)

    applications: list[list[Union[int, str]]] = Field(
        ..., description="List of [council_id (int), planning_reference (str)] pairs"
    )
    extensions: LookupExtensions = Field(default_factory=LookupExtensions)


# Endpoint 3 — Council stats


class StatsInput(BaseModel):
    """``input`` block for the council stats endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    council_id: int = Field(
        ..., description="Council identifier (integer)"
    )
    date_from: Optional[str] = Field(
        None, description="ISO-8601 start date"
    )
    date_to: Optional[str] = Field(
        None, description="ISO-8601 end date"
    )


class StatsRequest(BaseModel):
    """Full request body for the council stats endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    input: StatsInput


# ── Response wrappers ────────────────────────────────────────────────────────


class SearchResponse(BaseModel):
    """Envelope returned by the search endpoint.

    Note: the search endpoint may return a flat list of applications
    rather than a wrapper object.  The ``from_api_response`` class method
    handles both cases.
    """

    model_config = ConfigDict(populate_by_name=True)

    applications: list[PlanningApplication] = Field(default_factory=list)
    total_results: int = Field(0)
    page: int = Field(1)
    page_size: int = Field(100)

    @classmethod
    def from_api_response(
        cls, data: Any, page: int = 1, page_size: int = 100,
    ) -> SearchResponse:
        """Build from raw API JSON which may be a list or a dict."""
        if isinstance(data, list):
            apps = [PlanningApplication.model_validate(item) for item in data]
            return cls(
                applications=apps,
                total_results=len(apps),
                page=page,
                page_size=page_size,
            )
        return cls.model_validate(data)


class LookupResponse(BaseModel):
    """Envelope returned by the application lookup endpoint.

    Like the search endpoint, the API may return a flat list.
    """

    model_config = ConfigDict(populate_by_name=True)

    applications: list[PlanningApplication] = Field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: Any) -> LookupResponse:
        """Build from raw API JSON which may be a list or a dict."""
        if isinstance(data, list):
            apps = [PlanningApplication.model_validate(item) for item in data]
            return cls(applications=apps)
        return cls.model_validate(data)
