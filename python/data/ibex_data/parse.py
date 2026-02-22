"""Conversion functions between Ibex Pydantic models and DB row tuples.

Handles JSON serialisation of nested objects (unit mix, floor area, appeals,
document metadata) so they can be stored as TEXT blobs in SQLite and
reconstructed losslessly.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hashbrowns.ibex.models import ApplicationsResponse, StatsResponse


def _json_dump(obj) -> str | None:
    """Serialise a Pydantic model or list of models to a JSON string."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return json.dumps([item.model_dump() for item in obj])
    return json.dumps(obj.model_dump())


def _date_str(d) -> str | None:
    """Convert a date/datetime to an ISO string, or pass through strings."""
    if d is None:
        return None
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)


# ---------------------------------------------------------------------------
# Model → DB record
# ---------------------------------------------------------------------------


def application_to_record(app: ApplicationsResponse) -> tuple:
    """Flatten an ``ApplicationsResponse`` into an insert tuple for
    ``ibex_applications``."""
    return (
        app.council_id,
        app.council_name,
        app.planning_reference,
        app.url,
        app.proposal,
        app.raw_address,
        app.raw_application_type,
        app.normalised_application_type.value if app.normalised_application_type else None,
        app.normalised_decision.value if app.normalised_decision else None,
        app.raw_decision,
        _date_str(app.application_date),
        _date_str(app.decided_date),
        app.geometry,
        app.project_type.value if app.project_type else None,
        app.heading,
        app.num_new_houses,
        app.num_comments_received,
        _json_dump(app.proposed_unit_mix),
        _json_dump(app.proposed_floor_area),
        _json_dump(app.document_metadata),
        _json_dump(app.appeals),
        datetime.utcnow().isoformat(),
    )


def stats_to_record(
    stats: StatsResponse,
    council_id: int,
    date_from: str,
    date_to: str,
) -> tuple:
    """Flatten a ``StatsResponse`` into an insert tuple for
    ``ibex_council_stats``."""
    return (
        council_id,
        stats.approval_rate,
        stats.refusal_rate,
        json.dumps(stats.average_decision_time.model_dump(by_alias=True)),
        json.dumps(stats.number_of_applications.model_dump(by_alias=True)),
        stats.number_of_new_homes_approved,
        stats.council_development_activity_level.value,
        date_from,
        date_to,
        datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# DB record → Model
# ---------------------------------------------------------------------------


def _parse_json(text: str | None):
    """Parse a JSON text blob, returning None if empty."""
    if text is None:
        return None
    return json.loads(text)


def records_to_applications(rows: list[tuple]) -> list:
    """Reconstruct ``ApplicationsResponse`` Pydantic models from DB rows.

    Imports are deferred to avoid circular dependencies when this module is
    used from the ``python/data`` package (which doesn't depend on
    ``hashbrowns``).
    """
    from hashbrowns.ibex.models import ApplicationsResponse

    results = []
    for row in rows:
        # row indices match the SELECT * column order (skipping id at 0)
        data = {
            "council_id": row[1],
            "council_name": row[2],
            "planning_reference": row[3],
            "url": row[4],
            "proposal": row[5],
            "raw_address": row[6],
            "raw_application_type": row[7],
            "normalised_application_type": row[8],
            "normalised_decision": row[9],
            "raw_decision": row[10],
            "application_date": row[11],
            "decided_date": row[12],
            "geometry": row[13],
            "project_type": row[14],
            "heading": row[15],
            "num_new_houses": row[16],
            "num_comments_received": row[17],
            "proposed_unit_mix": _parse_json(row[18]),
            "proposed_floor_area": _parse_json(row[19]),
            "document_metadata": _parse_json(row[20]),
            "appeals": _parse_json(row[21]),
        }
        # Remove None values so Pydantic defaults kick in
        data = {k: v for k, v in data.items() if v is not None}
        results.append(ApplicationsResponse.model_validate(data))
    return results


def records_to_stats(rows: list[tuple]) -> list:
    """Reconstruct ``StatsResponse`` Pydantic models from DB rows."""
    from hashbrowns.ibex.models import StatsResponse

    results = []
    for row in rows:
        data = {
            "approval_rate": row[2],
            "refusal_rate": row[3],
            "average_decision_time": _parse_json(row[4]),
            "number_of_applications": _parse_json(row[5]),
            "number_of_new_homes_approved": row[6],
            "council_development_activity_level": row[7],
        }
        results.append(StatsResponse.model_validate(data))
    return results
