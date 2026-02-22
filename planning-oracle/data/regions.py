"""Region ↔ council mapping for the planning oracle.

Loads the canonical REGIONS / COUNCIL_REGION dicts from the shared
``python/data/regions.py`` (via importlib to avoid circular imports
with the local ``data`` package) and provides helpers to resolve
region from a ``council_id`` and to normalise parser region names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load python/data/regions.py by absolute path to avoid the circular
# import that occurs when ``from data.regions import ...`` resolves to
# this file (planning-oracle/data/regions.py) instead of the canonical
# python/data/regions.py.
_CANONICAL = Path(__file__).resolve().parents[2] / "python" / "data" / "regions.py"
_spec = importlib.util.spec_from_file_location("_canonical_regions", _CANONICAL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REGIONS: dict[str, list[str]] = _mod.REGIONS
COUNCIL_REGION: dict[str, str] = _mod.COUNCIL_REGION

__all__ = [
    "COUNCIL_REGION",
    "REGIONS",
    "get_region_for_council",
    "get_council_ids_for_region",
    "normalise_region_name",
    "resolve_council_region",
]

# Map parser / colloquial region names → canonical region names used in
# python/data/regions.py.  The parser emits names like "Greater London"
# or "Midlands" which don't exist in the canonical set.
_REGION_ALIASES: dict[str, str | None] = {
    "greater london": "London",
    "south england": "South East",
    "midlands": "West Midlands",
    "yorkshire": "Yorkshire & Humber",
    "wales": None,
    "scotland": None,
}


# Map council names that differ between the Ibex DB and the canonical
# COUNCIL_REGION dictionary.  Keys are lowered Ibex names, values are
# the canonical COUNCIL_REGION key.
_COUNCIL_NAME_ALIASES: dict[str, str] = {
    "royal greenwich": "Greenwich",
    "birmingham city": "Birmingham",
    "leicester city": "Leicester",
    "milton-keynes": "Milton Keynes",
    "st helens": "St Helens",  # may need adding to canonical set
    "adur and worthing": "Adur",  # merged council → use first
    "babergh and mid-suffolk": "Babergh",  # merged council → use first
    "north northamptonshire": "North Northamptonshire",
}


def resolve_council_region(council_name: str | None) -> str | None:
    """Resolve a council name to its canonical region.

    Handles exact matches, known aliases, and simple fuzzy matching
    (hyphen/space normalisation).
    """
    if not council_name:
        return None

    # Exact match.
    region = COUNCIL_REGION.get(council_name)
    if region:
        return region

    # Try alias table.
    alias = _COUNCIL_NAME_ALIASES.get(council_name.lower())
    if alias:
        return COUNCIL_REGION.get(alias)

    # Normalise hyphens → spaces and retry.
    normalised = council_name.replace("-", " ")
    region = COUNCIL_REGION.get(normalised)
    if region:
        return region

    return None


def normalise_region_name(raw: str | None) -> str | None:
    """Convert a parser-emitted region name to a canonical one.

    Returns ``None`` if the region cannot be mapped (e.g. Wales/Scotland
    which are outside the English region set).
    """
    if raw is None:
        return None

    # Already canonical?
    if raw in REGIONS:
        return raw

    lowered = raw.strip().lower()

    # Check alias table.
    if lowered in _REGION_ALIASES:
        return _REGION_ALIASES[lowered]

    # Fuzzy: check if any canonical name starts with or contains the input.
    for canonical in REGIONS:
        if canonical.lower().startswith(lowered) or lowered in canonical.lower():
            return canonical

    return None


def get_region_for_council(
    council_id: int,
    council_stats: dict,
) -> str | None:
    """Look up the canonical region for a council by its ID.

    Uses the ``council_name`` field from the stats dict to resolve via
    ``COUNCIL_REGION``.
    """
    stats = council_stats.get(council_id)
    if stats is None:
        return None
    name = getattr(stats, "council_name", None)
    if name is None:
        return None
    return COUNCIL_REGION.get(name)


def get_council_ids_for_region(
    region: str,
    council_stats: dict,
) -> list[int]:
    """Return council IDs from ``council_stats`` that belong to *region*.

    Returns an empty list if no councils match.
    """
    region_councils = set(REGIONS.get(region, []))
    if not region_councils:
        return []

    matched: list[int] = []
    for cid, stats in council_stats.items():
        name = getattr(stats, "council_name", None)
        if name and name in region_councils:
            matched.append(cid)
    return matched
