"""Location enrichment utilities.

Resolves raw location text into council names, coordinates, and search
radii.  Mutates the ``LocationIntent`` on a ``ParsedIntent`` in place.

TODO: implement — this is the scaffold.
"""

from __future__ import annotations

from .schema import LocationIntent


async def enrich_location(location: LocationIntent) -> LocationIntent:
    """Geocode and resolve council names for a LocationIntent.

    Takes the raw location fields (``raw_text``, ``names``, ``level``) and
    populates ``resolved_councils``, ``resolved_coordinates``, and
    ``radius_suggestion_m``.

    Args:
        location: A LocationIntent with at least ``raw_text`` populated.

    Returns:
        The same LocationIntent, enriched with resolved fields.
    """
    raise NotImplementedError("Location enrichment not yet implemented")
