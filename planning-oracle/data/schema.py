"""Pydantic models matching the Planning API request and response shapes.

Each model mirrors the JSON structure of a specific API endpoint.  Field
aliases handle the camelCase ↔ snake_case conversion so that Python code
uses idiomatic names while the wire format stays compatible with the API.

All models set ``populate_by_name=True`` so instances can be constructed with
either the Python name or the JSON alias.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Nested / reusable models ────────────────────────────────────────────────


class ProposedUnitMix(BaseModel):
    """Breakdown of proposed residential units by bedroom count."""

    model_config = ConfigDict(populate_by_name=True)

    one_bed: int = Field(0, alias="oneBed", description="Number of 1-bedroom units")
    two_bed: int = Field(0, alias="twoBed", description="Number of 2-bedroom units")
    three_bed: int = Field(
        0, alias="threeBed", description="Number of 3-bedroom units"
    )
    four_plus_bed: int = Field(
        0, alias="fourPlusBed", description="Number of 4+ bedroom units"
    )
    affordable: int = Field(0, description="Number of affordable housing units")


class ProposedFloorArea(BaseModel):
    """Proposed floor area details for the application."""

    model_config = ConfigDict(populate_by_name=True)

    gross_sqm: float = Field(
        0.0, alias="grossSqm", description="Gross internal area in square metres"
    )
    net_sqm: float = Field(
        0.0, alias="netSqm", description="Net internal area in square metres"
    )
    use_class: str = Field(
        "", alias="useClass", description="Planning use class (e.g. C3, B1)"
    )


class DocumentMetadata(BaseModel):
    """Metadata for a document attached to a planning application."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(
        ..., alias="documentId", description="Unique identifier for the document"
    )
    title: str = Field("", description="Document title")
    document_type: str = Field(
        "", alias="documentType", description="Type/category of document"
    )
    url: str = Field("", description="URL to retrieve the document (may be S3 link)")
    date_published: Optional[date] = Field(
        None, alias="datePublished", description="Date the document was published"
    )


# ── Main data models ────────────────────────────────────────────────────────


class PlanningApplication(BaseModel):
    """Planning application record returned by the search and lookup endpoints.

    This is the primary unit of data in the pipeline.  Each record captures
    the proposal description, site details, decision outcome, and optional
    nested objects for unit mix, floor area, and attached documents.

    Fields are nullable because different endpoints populate different subsets.
    """

    model_config = ConfigDict(populate_by_name=True)

    application_id: str = Field(
        ..., alias="applicationId", description="Unique application reference"
    )
    council_id: str = Field(
        ..., alias="councilId", description="Local planning authority identifier"
    )
    council_name: Optional[str] = Field(
        None, alias="councilName", description="Name of the local planning authority"
    )
    planning_reference: Optional[str] = Field(
        None,
        alias="planningReference",
        description="Human-readable planning reference (e.g. 24/00123/FUL)",
    )
    description: Optional[str] = Field(
        None, description="Free-text proposal description"
    )
    address: Optional[str] = Field(None, description="Site address")
    postcode: Optional[str] = Field(None, description="Site postcode")
    ward: Optional[str] = Field(None, description="Electoral ward")
    application_type: Optional[str] = Field(
        None,
        alias="applicationType",
        description="Application type (e.g. Full, Outline, Reserved Matters)",
    )
    normalised_application_type: Optional[str] = Field(
        None,
        alias="normalisedApplicationType",
        description="Normalised application type for filtering",
    )
    project_type: Optional[str] = Field(
        None,
        alias="projectType",
        description="High-level project category (e.g. residential, commercial)",
    )
    heading: Optional[str] = Field(
        None, description="Short heading summarising the proposal"
    )
    status: Optional[str] = Field(
        None, description="Current status (e.g. Pending, Decided)"
    )
    decision: Optional[str] = Field(
        None, description="Decision outcome if determined"
    )
    normalised_decision: Optional[str] = Field(
        None,
        alias="normalisedDecision",
        description="Normalised decision value for filtering",
    )
    decision_date: Optional[date] = Field(
        None, alias="decisionDate", description="Date of decision"
    )
    date_received: Optional[date] = Field(
        None, alias="dateReceived", description="Date application received"
    )
    date_validated: Optional[date] = Field(
        None, alias="dateValidated", description="Date application validated"
    )
    proposed_units: Optional[ProposedUnitMix] = Field(
        None, alias="proposedUnits", description="Proposed unit mix if residential"
    )
    proposed_floor_area: Optional[ProposedFloorArea] = Field(
        None, alias="proposedFloorArea", description="Proposed floor areas"
    )
    documents: list[DocumentMetadata] = Field(
        default_factory=list, description="Attached documents"
    )
    num_new_houses: Optional[int] = Field(
        None,
        alias="numNewHouses",
        description="Number of net new houses proposed",
    )


class CouncilStats(BaseModel):
    """Aggregated statistics for a local planning authority.

    Returned by the stats endpoint.  Contains approval/refusal rates,
    decision speed broken down by project type, application counts, and
    an activity-level classification.
    """

    model_config = ConfigDict(populate_by_name=True)

    council_id: str = Field(
        ..., alias="councilId", description="Local planning authority identifier"
    )
    council_name: Optional[str] = Field(
        None, alias="councilName", description="Name of the local planning authority"
    )
    approval_rate: Optional[float] = Field(
        None, alias="approvalRate", description="Historical approval rate (0-1)"
    )
    refusal_rate: Optional[float] = Field(
        None, alias="refusalRate", description="Historical refusal rate (0-1)"
    )
    average_decision_time: Optional[dict[str, float]] = Field(
        None,
        alias="averageDecisionTime",
        description="Average days to decision keyed by project type",
    )
    number_of_applications: Optional[dict[str, int]] = Field(
        None,
        alias="numberOfApplications",
        description="Application counts keyed by type",
    )
    number_of_new_homes_approved: Optional[int] = Field(
        None,
        alias="numberOfNewHomesApproved",
        description="Total net new homes approved in the period",
    )
    council_development_activity_level: Optional[str] = Field(
        None,
        alias="councilDevelopmentActivityLevel",
        description="Activity classification (e.g. high, medium, low)",
    )
    period_start: Optional[date] = Field(
        None, alias="periodStart", description="Stats period start date"
    )
    period_end: Optional[date] = Field(
        None, alias="periodEnd", description="Stats period end date"
    )


class ApplicationDocument(BaseModel):
    """Full document content retrieved from the document lookup endpoint.

    Extends ``DocumentMetadata`` with the actual text content extracted from
    the document file (e.g. via OCR or PDF parsing on the API side).
    """

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(
        ..., alias="documentId", description="Unique identifier for the document"
    )
    application_id: str = Field(
        ..., alias="applicationId", description="Parent application reference"
    )
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    content_text: Optional[str] = Field(
        None, alias="contentText", description="Extracted text content of the document"
    )
    content_url: Optional[str] = Field(
        None, alias="contentUrl", description="Direct URL to the document file"
    )


# ── Request body models ─────────────────────────────────────────────────────

# Endpoint 1 — Search applications


class SearchInput(BaseModel):
    """``input`` block for the search endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    date_range_type: str = Field(
        "determined",
        alias="dateRangeType",
        description="Which date field to filter on (e.g. determined, validated)",
    )
    date_from: Optional[str] = Field(
        None, alias="dateFrom", description="ISO-8601 start date"
    )
    date_to: Optional[str] = Field(
        None, alias="dateTo", description="ISO-8601 end date"
    )
    council_id: Optional[str] = Field(
        None, alias="councilId", description="Council identifier (mutually exclusive with coordinates)"
    )
    coordinates: Optional[list[float]] = Field(
        None, description="[lon, lat] for radius search"
    )
    radius: Optional[float] = Field(
        None, description="Search radius (units determined by srid)"
    )
    srid: Optional[int] = Field(
        None, description="Spatial reference ID for the coordinates"
    )
    page: int = Field(1, description="Page number (1-based)")
    page_size: int = Field(100, alias="pageSize", description="Results per page")


class SearchExtensions(BaseModel):
    """``extensions`` block for the search endpoint — all boolean toggles."""

    model_config = ConfigDict(populate_by_name=True)

    documents: bool = Field(False, description="Include attached documents")
    appeals: bool = Field(False, description="Include appeal information")
    project_type: bool = Field(
        False, alias="projectType", description="Include project type classification"
    )
    heading: bool = Field(False, description="Include proposal heading")


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
        alias="normalisedApplicationType",
        description="Filter by normalised application types",
    )
    project_type: list[str] = Field(
        default_factory=list,
        alias="projectType",
        description="Filter by project types",
    )
    normalised_decision: list[str] = Field(
        default_factory=list,
        alias="normalisedDecision",
        description="Filter by normalised decisions",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Free-text keyword filters"
    )
    num_new_houses: Optional[NumNewHousesFilter] = Field(
        None,
        alias="numNewHouses",
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
    """``extensions`` block for the lookup endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    documents: bool = Field(False, description="Include documents with S3 links")
    appeals: bool = Field(False, description="Include appeal information")
    project_type: bool = Field(
        False, alias="projectType", description="Include project type"
    )
    heading: bool = Field(False, description="Include proposal heading")


class LookupRequest(BaseModel):
    """Full request body for the application lookup endpoint.

    ``applications`` is a list of ``[council_id, planning_reference]`` pairs.
    """

    model_config = ConfigDict(populate_by_name=True)

    applications: list[list[str]] = Field(
        ..., description="List of [councilId, planningReference] pairs"
    )
    extensions: LookupExtensions = Field(default_factory=LookupExtensions)


# Endpoint 3 — Council stats


class StatsInput(BaseModel):
    """``input`` block for the council stats endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    council_id: str = Field(
        ..., alias="councilId", description="Council identifier"
    )
    date_from: Optional[str] = Field(
        None, alias="dateFrom", description="ISO-8601 start date"
    )
    date_to: Optional[str] = Field(
        None, alias="dateTo", description="ISO-8601 end date"
    )


class StatsRequest(BaseModel):
    """Full request body for the council stats endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    input: StatsInput


# ── Response wrappers ────────────────────────────────────────────────────────


class SearchResponse(BaseModel):
    """Envelope returned by the search endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    applications: list[PlanningApplication] = Field(default_factory=list)
    total_results: int = Field(0, alias="totalResults")
    page: int = Field(1)
    page_size: int = Field(100, alias="pageSize")


class LookupResponse(BaseModel):
    """Envelope returned by the application lookup endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    applications: list[PlanningApplication] = Field(default_factory=list)
