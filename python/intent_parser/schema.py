"""Canonical intent schema for the planning site finder.

Pure domain models describing what a user wants — zero coupling to any
external API.  These models are the shared language between the NLP parser,
the enrichment pipeline, and any downstream API adapters.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class DevelopmentIntent(BaseModel):
    """What the user wants to build, convert, or demolish."""

    category: str = Field(
        description=(
            "Normalised development category, e.g. "
            '"residential", "commercial", "mixed_use", "hospitality", '
            '"change_of_use", "home_improvement", "infrastructure".'
        ),
    )
    subcategory: Optional[str] = Field(
        default=None,
        description=(
            "More specific type, e.g. "
            '"affordable_housing", "HMO", "student_accommodation", '
            '"hotel", "warehouse_conversion", "loft_extension", "rear_extension".'
        ),
    )
    description: str = Field(
        description='Normalised plain-English summary, e.g. "20-unit affordable housing block".',
    )
    scale: Optional[str] = Field(
        default=None,
        description=(
            "Planning-scale classification: "
            '"minor" (<10 units), "major" (10+), "large_major" (100+).'
        ),
    )
    unit_count: Optional[int] = Field(
        default=None,
        description="Number of units / dwellings if mentioned.",
    )
    use_class: Optional[str] = Field(
        default=None,
        description=(
            "UK planning use class if inferable: "
            '"C3" (residential), "E" (commercial), "F1" (learning), "sui_generis", etc.'
        ),
    )
    from_use: Optional[str] = Field(
        default=None,
        description="For change-of-use: the current use class or description.",
    )
    to_use: Optional[str] = Field(
        default=None,
        description="For change-of-use: the desired use class or description.",
    )
    raw_tags: list[str] = Field(
        default_factory=list,
        description='Descriptive tags extracted from the query: ["affordable", "social", "eco", "modular"].',
    )


class LocationIntent(BaseModel):
    """Where the user wants to build or search."""

    raw_text: str = Field(
        description='Exactly what the user said about location, e.g. "South London".',
    )
    level: str = Field(
        default="unspecified",
        description=(
            "Granularity of the location reference: "
            '"address", "neighbourhood", "borough", "city", "region", '
            '"county", "country", "unspecified".'
        ),
    )
    names: list[str] = Field(
        default_factory=list,
        description='Parsed location names: ["South London"] or ["Hackney"] or ["Shoreditch"].',
    )
    resolved_councils: list[str] = Field(
        default_factory=list,
        description=(
            "Borough / council names resolved during enrichment, "
            'e.g. ["Lambeth", "Southwark"]. Empty until enrichment.'
        ),
    )
    resolved_coordinates: Optional[dict] = Field(
        default=None,
        description='WGS84 coordinates once geocoded: {"lat": 51.5, "lng": -0.1}. None until enrichment.',
    )
    radius_suggestion_m: Optional[int] = Field(
        default=None,
        description="Suggested search radius in metres, inferred from location level.",
    )
    country: str = Field(
        default="England",
        description='Country context: "England", "Wales", "Scotland", etc.',
    )


class Constraint(BaseModel):
    """A single constraint the user expressed — something to avoid, require, or prefer."""

    type: str = Field(
        description='"avoid", "require", or "prefer".',
    )
    category: str = Field(
        description=(
            "What the constraint is about: "
            '"flood_risk", "conservation_area", "green_belt", "article_4", '
            '"listed_building", "epc_rating", "budget", "timeline", '
            '"transport_links", "schools_nearby", etc.'
        ),
    )
    value: Optional[str] = Field(
        default=None,
        description='Specific value, e.g. "zone_3" for flood risk, "A" for EPC, "under_500k" for budget.',
    )
    raw_text: str = Field(
        description="What the user actually said for this constraint.",
    )


class AnalysisGoal(BaseModel):
    """What the user actually wants to learn or achieve."""

    goal: str = Field(
        description=(
            "The type of analysis: "
            '"find_sites", "check_feasibility", "compare_areas", '
            '"understand_refusals", "track_trends", "assess_risk", "explore".'
        ),
    )
    detail: Optional[str] = Field(
        default=None,
        description='Extra context, e.g. "which borough has highest approval rate for HMOs".',
    )
    time_range: Optional[dict] = Field(
        default=None,
        description='Date range if relevant: {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}.',
    )


# ---------------------------------------------------------------------------
# Top-level parsed intent
# ---------------------------------------------------------------------------


class ParsedIntent(BaseModel):
    """Top-level output of the NLP intent parser.

    Represents the fully structured, API-agnostic interpretation of a user's
    natural language planning query.
    """

    id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique identifier for this parse result.",
    )
    raw_query: str = Field(
        description="The original user input, verbatim.",
    )
    development: DevelopmentIntent
    location: LocationIntent
    constraints: list[Constraint] = Field(default_factory=list)
    analysis_goals: list[AnalysisGoal] = Field(default_factory=list)
    keywords: list[str] = Field(
        default_factory=list,
        description="Freeform keywords extracted from the query.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident the parser is in this interpretation (0–1).",
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Things the parser wasn't sure about — useful for follow-up questions.",
    )

    def to_dict(self) -> dict:
        """Serialise the entire intent to a plain JSON-compatible dict."""
        return self.model_dump(mode="json")

    def to_summary(self) -> str:
        """Return a human-readable 2–3 line summary of the parsed intent."""
        parts: list[str] = []

        # Line 1 — what + where
        what = self.development.description
        where = self.location.raw_text or "unspecified location"
        parts.append(f"{what} in {where}.")

        # Line 2 — constraints (if any)
        if self.constraints:
            constraint_strs = [
                f"{c.type} {c.category}" + (f" ({c.value})" if c.value else "")
                for c in self.constraints
            ]
            parts.append("Constraints: " + ", ".join(constraint_strs) + ".")

        # Line 3 — goals (if any)
        if self.analysis_goals:
            goal_strs = [
                g.goal.replace("_", " ") + (f" — {g.detail}" if g.detail else "")
                for g in self.analysis_goals
            ]
            parts.append("Goals: " + "; ".join(goal_strs) + ".")

        return "\n".join(parts)
