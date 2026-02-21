"""Pydantic v2 models mirroring the Ibex Enterprise API OpenAPI schema.

All models use model_config extra="ignore" to tolerate undocumented fields
gracefully. Date fields with ISO datetime-with-timezone strings (e.g.
"2025-04-10T00:00:00.000Z") are coerced to plain date objects via a
field_validator on BaseApplicationsSchema.

Hierarchy:
  Enums
    NormalisedDecision
    NormalisedApplicationType
    ProjectType
    CouncilDevelopmentActivityLevel

  Sub-models (nested within application records)
    UnitBreakdownItemSchema
    AppealsSchema
    ProposedUnitMixSchema
    ProposedFloorAreaSchema
    DocumentMetadataItemSchema

  Stats sub-models
    AverageDecisionTime       -- spaced/hyphenated JSON keys aliased to snake_case
    NumberOfApplications      -- same alias pattern

  Application response models
    BaseApplicationsSchema    -- shared required + optional base fields
    SearchResponse            -- extends Base with 9 extension fields incl. centre_point
    ApplicationsResponse      -- extends Base with same extensions MINUS centre_point

  Stats response model
    StatsResponse             -- standalone; required: all 6 top-level fields
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NormalisedDecision(str, Enum):
    """Possible normalised decision outcomes for a planning application."""

    Approved = "Approved"
    Other = "Other"
    Refused = "Refused"
    Unknown = "Unknown"
    Validated = "Validated"
    Withdrawn = "Withdrawn"


class NormalisedApplicationType(str, Enum):
    """Normalised application type categories."""

    non_material_amendment = "non-material amendment"
    discharge_of_conditions = "discharge of conditions"
    listed_building_consent = "listed building consent"
    advertisement_consent = "advertisement consent"
    householder_planning_application = "householder planning application"
    tree_preservation_order = "tree preservation order"
    lawful_development = "lawful development"
    change_of_use = "change of use"
    full_planning_application = "full planning application"
    conservation_area = "conservation area"
    utilities = "utilities"
    unknown = "unknown"
    environmental_impact = "environmental impact"
    section_106 = "section 106"
    pre_application = "pre-application"
    other = "other"


class ProjectType(str, Enum):
    """Project type categories used by the Ibex classifier."""

    small_residential = "small residential"
    tree = "tree"
    large_residential = "large residential"
    home_improvement = "home improvement"
    mixed = "mixed"
    medium_residential = "medium residential"


class CouncilDevelopmentActivityLevel(str, Enum):
    """Development activity level for a council."""

    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class UnitBreakdownItemSchema(BaseModel):
    """Individual unit type/tenure/bedroom combination within a proposed unit mix."""

    model_config = ConfigDict(extra="ignore")

    residential_unit_type: Optional[str] = None
    tenure: Optional[str] = None
    development_type: Optional[str] = None
    number_of_units: Optional[int] = None
    bedrooms_per_unit: Optional[int] = None
    provider: Optional[str] = None
    gia_per_unit: Optional[float] = None


class AppealsSchema(BaseModel):
    """Details about planning appeals associated with applications."""

    model_config = ConfigDict(extra="ignore")

    appeal_ref: str
    appeal_url: str
    start_date: Optional[str] = None
    decision_date: Optional[str] = None
    decision: Optional[str] = None
    case_type: Optional[str] = None


class ProposedUnitMixSchema(BaseModel):
    """Residential unit breakdown from extracted application forms."""

    model_config = ConfigDict(extra="ignore")

    total_existing_residential_units: Optional[int] = None
    total_proposed_residential_units: Optional[int] = None
    proposed_1_bed_units: Optional[int] = None
    proposed_2_bed_units: Optional[int] = None
    proposed_3_bed_units: Optional[int] = None
    proposed_4_plus_bed_units: Optional[int] = None
    affordable_housing_units: Optional[int] = None
    proposed_terraced_count: Optional[int] = None
    proposed_semi_count: Optional[int] = None
    proposed_detached_count: Optional[int] = None
    proposed_flat_count: Optional[int] = None
    proposed_market_units: Optional[int] = None
    proposed_affordable_units: Optional[int] = None
    proposed_shared_ownership_units: Optional[int] = None
    proposed_social_rent_units: Optional[int] = None
    units_to_be_added: Optional[List[UnitBreakdownItemSchema]] = None
    units_to_be_lost: Optional[List[UnitBreakdownItemSchema]] = None


class ProposedFloorAreaSchema(BaseModel):
    """Floor area data from extracted application forms."""

    model_config = ConfigDict(extra="ignore")

    gross_internal_area_to_add_sqm: Optional[float] = None
    existing_gross_floor_area_sqm: Optional[float] = None
    proposed_gross_floor_area_sqm: Optional[float] = None
    floor_area_to_be_lost_sqm: Optional[float] = None
    floor_area_to_be_gained_sqm: Optional[float] = None


class DocumentMetadataItemSchema(BaseModel):
    """Metadata about a document associated with a planning application."""

    model_config = ConfigDict(extra="ignore")

    date_published: Optional[str] = None
    document_type: Optional[str] = None
    description: Optional[str] = None
    document_link: Optional[str] = None


# ---------------------------------------------------------------------------
# Stats sub-models — JSON keys contain spaces/hyphens; aliased to snake_case
# ---------------------------------------------------------------------------


class AverageDecisionTime(BaseModel):
    """Average decision time in days for each project type.

    JSON keys use spaces (e.g. "small residential"); Python fields use
    underscores with Field(alias=...) and populate_by_name=True so tests can
    also construct using snake_case Python identifiers.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    small_residential: Optional[float] = Field(None, alias="small residential")
    tree: Optional[float] = None
    large_residential: Optional[float] = Field(None, alias="large residential")
    home_improvement: Optional[float] = Field(None, alias="home improvement")
    mixed: Optional[float] = None
    medium_residential: Optional[float] = Field(None, alias="medium residential")


class NumberOfApplications(BaseModel):
    """Count of applications for each normalised application type.

    JSON keys use hyphens and spaces; Python fields use underscores with
    aliases. populate_by_name=True allows construction via snake_case names.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    non_material_amendment: Optional[int] = Field(None, alias="non-material amendment")
    discharge_of_conditions: Optional[int] = Field(None, alias="discharge of conditions")
    listed_building_consent: Optional[int] = Field(None, alias="listed building consent")
    advertisement_consent: Optional[int] = Field(None, alias="advertisement consent")
    householder_planning_application: Optional[int] = Field(
        None, alias="householder planning application"
    )
    tree_preservation_order: Optional[int] = Field(None, alias="tree preservation order")
    lawful_development: Optional[int] = Field(None, alias="lawful development")
    change_of_use: Optional[int] = Field(None, alias="change of use")
    full_planning_application: Optional[int] = Field(None, alias="full planning application")
    conservation_area: Optional[int] = Field(None, alias="conservation area")
    utilities: Optional[int] = None
    unknown: Optional[int] = None
    environmental_impact: Optional[int] = Field(None, alias="environmental impact")
    section_106: Optional[int] = Field(None, alias="section 106")
    pre_application: Optional[int] = Field(None, alias="pre-application")
    other: Optional[int] = None


# ---------------------------------------------------------------------------
# Base application model
# ---------------------------------------------------------------------------


class BaseApplicationsSchema(BaseModel):
    """Core planning application data shared across all response types.

    Required fields: council_id, council_name, planning_reference, url,
    normalised_decision, geometry. All other fields are Optional with None
    default.

    Date fields handle both ISO date strings ("2025-04-01") and ISO datetime-
    with-timezone strings ("2025-04-10T00:00:00.000Z") by stripping the time
    component via a field_validator.
    """

    model_config = ConfigDict(extra="ignore")

    # Required fields
    council_id: int
    council_name: str
    planning_reference: str
    url: Optional[str] = None
    normalised_decision: NormalisedDecision
    geometry: Optional[str] = None

    # Optional base fields
    proposal: Optional[str] = None
    raw_address: Optional[str] = None
    raw_application_type: Optional[str] = None
    normalised_application_type: Optional[NormalisedApplicationType] = None
    application_date: Optional[date] = None
    decided_date: Optional[date] = None
    raw_decision: Optional[str] = None

    @field_validator("application_date", "decided_date", mode="before")
    @classmethod
    def coerce_date(cls, v: object) -> object:
        """Strip time/timezone from ISO datetime strings, returning a plain date.

        Handles:
        - None → None (pass through)
        - "2025-04-01" → date(2025, 4, 1) (Pydantic handles this natively)
        - "2025-04-10T00:00:00.000Z" → date(2025, 4, 10) (stripped here)
        - date/datetime objects → passed through as-is
        """
        if v is None:
            return None
        if isinstance(v, str) and "T" in v:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
        return v


# ---------------------------------------------------------------------------
# Search endpoint response model
# ---------------------------------------------------------------------------


class SearchResponse(BaseApplicationsSchema):
    """Response item from the /search endpoint.

    Extends BaseApplicationsSchema with 9 optional extension fields including
    centre_point (not present in ApplicationsResponse).
    """

    # Extension fields — all Optional, all default to None
    appeals: Optional[List[AppealsSchema]] = None
    project_type: Optional[ProjectType] = None
    centre_point: Optional[str] = None
    heading: Optional[str] = None
    num_new_houses: Optional[int] = None
    document_metadata: Optional[List[DocumentMetadataItemSchema]] = None
    proposed_unit_mix: Optional[ProposedUnitMixSchema] = None
    proposed_floor_area: Optional[ProposedFloorAreaSchema] = None
    num_comments_received: Optional[int] = None


# ---------------------------------------------------------------------------
# Applications endpoint response model
# ---------------------------------------------------------------------------


class ApplicationsResponse(BaseApplicationsSchema):
    """Response item from the /applications endpoint.

    Extends BaseApplicationsSchema with the same extension fields as
    SearchResponse EXCEPT centre_point, which is absent from the
    ApplicationsResponseSchema in the OpenAPI spec.
    """

    appeals: Optional[List[AppealsSchema]] = None
    project_type: Optional[ProjectType] = None
    heading: Optional[str] = None
    num_new_houses: Optional[int] = None
    document_metadata: Optional[List[DocumentMetadataItemSchema]] = None
    proposed_unit_mix: Optional[ProposedUnitMixSchema] = None
    proposed_floor_area: Optional[ProposedFloorAreaSchema] = None
    num_comments_received: Optional[int] = None


# ---------------------------------------------------------------------------
# Stats endpoint response model
# ---------------------------------------------------------------------------


class StatsResponse(BaseModel):
    """Council-level planning statistics from the /stats endpoint.

    All six fields are required per the OpenAPI spec. Uses extra="ignore" to
    tolerate any undocumented extension fields returned by the JWT-enriched
    response.
    """

    model_config = ConfigDict(extra="ignore")

    council_development_activity_level: CouncilDevelopmentActivityLevel
    approval_rate: float
    refusal_rate: float
    average_decision_time: AverageDecisionTime
    number_of_applications: NumberOfApplications
    number_of_new_homes_approved: int
