"""External council data: Housing Delivery Test and Green Belt.

Loads two government datasets and resolves them to council IDs via the
ibex.db name lookup.  Results are cached to ``data/external_council_data.json``
so subsequent calls skip the heavy parsing.

Data sources
~~~~~~~~~~~~
* Housing Delivery Test 2023 (``python/data/housing_test_data/2023.ods``)
* Green Belt designations (``python/data/green_belt_data/green-belt.geojson``)
* Council name↔ID mapping (``python/data/ibex_data/ibex.db``)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths relative to the repository root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # planning-oracle -> hl26-hashbrowns
_IBEX_DB = _REPO_ROOT / "python" / "data" / "ibex_data" / "ibex.db"
_HDT_ODS = _REPO_ROOT / "python" / "data" / "housing_test_data" / "2023.ods"
_GREEN_BELT_GEOJSON = _REPO_ROOT / "python" / "data" / "green_belt_data" / "green-belt.geojson"

_CACHE_PATH = Path(__file__).resolve().parent / "external_council_data.json"


@dataclass
class ExternalCouncilData:
    """External enrichment data for a single council."""

    hdt_measurement: Optional[float] = None
    has_green_belt: bool = False


# ── Name normalisation ────────────────────────────────────────────────────────

# Common suffixes/qualifiers that differ between datasets.
_STRIP_SUFFIXES = [
    " council",
    " borough council",
    " district council",
    " city council",
    " metropolitan borough council",
    " metropolitan district council",
    " london borough",
    " unitary authority",
    ", city of",
    " city",
]


def _normalise_name(name: str) -> str:
    """Normalise a council name for fuzzy matching.

    Lowercases, strips common suffixes, removes punctuation, and collapses
    whitespace.
    """
    s = name.strip().lower()
    for suffix in _STRIP_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Remove "the " prefix.
    if s.startswith("the "):
        s = s[4:]
    # Normalise punctuation and whitespace.
    s = s.replace("-", " ").replace(".", "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── ibex.db council lookup ────────────────────────────────────────────────────


def _build_council_lookup(db_path: Path = _IBEX_DB) -> dict[str, int]:
    """Build a normalised_name → council_id mapping from ibex.db."""
    if not db_path.exists():
        logger.warning("ibex.db not found at %s", db_path)
        return {}

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT council_id, council_name FROM ibex_applications "
            "WHERE council_name IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    lookup: dict[str, int] = {}
    for council_id, council_name in rows:
        norm = _normalise_name(council_name)
        lookup[norm] = int(council_id)
    return lookup


def _resolve_name(
    name: str,
    lookup: dict[str, int],
) -> Optional[int]:
    """Resolve a council name to an ID, trying exact then substring match."""
    norm = _normalise_name(name)

    # Exact match.
    if norm in lookup:
        return lookup[norm]

    # Try matching where one name contains the other.
    for known_name, cid in lookup.items():
        if norm in known_name or known_name in norm:
            return cid

    return None


# ── HDT parsing ───────────────────────────────────────────────────────────────


def _parse_hdt(
    ods_path: Path = _HDT_ODS,
    lookup: Optional[dict[str, int]] = None,
) -> dict[int, float]:
    """Parse the Housing Delivery Test ODS file.

    Returns:
        council_id → HDT measurement (raw percentage, e.g. 81.0 for 81%).
    """
    if not ods_path.exists():
        logger.warning("HDT ODS file not found at %s", ods_path)
        return {}

    import pandas as pd

    if lookup is None:
        lookup = _build_council_lookup()

    df = pd.read_excel(ods_path, engine="odf", header=None)
    # Data starts at row 6 (rows 0-5 are headers).
    data = df.iloc[6:].reset_index(drop=True)

    result: dict[int, float] = {}
    unmatched: list[str] = []

    for _, row in data.iterrows():
        area_name = row[1]
        hdt_value = row[10]

        if pd.isna(area_name):
            continue

        area_name_str = str(area_name).strip()
        council_id = _resolve_name(area_name_str, lookup)

        if council_id is None:
            unmatched.append(area_name_str)
            continue

        try:
            result[council_id] = float(hdt_value)
        except (TypeError, ValueError):
            pass

    if unmatched:
        logger.debug(
            "HDT: %d councils unmatched: %s",
            len(unmatched),
            unmatched[:10],
        )
    logger.info("HDT: resolved %d councils", len(result))
    return result


# ── Green Belt parsing ────────────────────────────────────────────────────────


def _parse_green_belt(
    geojson_path: Path = _GREEN_BELT_GEOJSON,
    lookup: Optional[dict[str, int]] = None,
) -> set[int]:
    """Parse the Green Belt GeoJSON file.

    Returns:
        Set of council_ids that have green belt land.
    """
    if not geojson_path.exists():
        logger.warning("Green Belt GeoJSON not found at %s", geojson_path)
        return set()

    if lookup is None:
        lookup = _build_council_lookup()

    with open(geojson_path) as f:
        data = json.load(f)

    council_ids_with_green_belt: set[int] = set()
    seen_names: set[str] = set()
    unmatched: list[str] = []

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name", "")
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        council_id = _resolve_name(name, lookup)
        if council_id is not None:
            council_ids_with_green_belt.add(council_id)
        else:
            unmatched.append(name)

    if unmatched:
        logger.debug(
            "Green Belt: %d councils unmatched: %s",
            len(unmatched),
            unmatched[:10],
        )
    logger.info("Green Belt: resolved %d councils", len(council_ids_with_green_belt))
    return council_ids_with_green_belt


# ── Public API ────────────────────────────────────────────────────────────────


def _build_external_data() -> dict[int, ExternalCouncilData]:
    """Parse source files and build the full external enrichment dict."""
    lookup = _build_council_lookup()
    hdt_map = _parse_hdt(lookup=lookup)
    green_belt_ids = _parse_green_belt(lookup=lookup)

    # Merge into a single dict keyed by council_id.
    all_ids = set(hdt_map.keys()) | green_belt_ids
    result: dict[int, ExternalCouncilData] = {}
    for cid in all_ids:
        result[cid] = ExternalCouncilData(
            hdt_measurement=hdt_map.get(cid),
            has_green_belt=cid in green_belt_ids,
        )

    logger.info(
        "External data: %d councils total (%d with HDT, %d with green belt)",
        len(result),
        len(hdt_map),
        len(green_belt_ids),
    )
    return result


def _save_cache(data: dict[int, ExternalCouncilData]) -> None:
    """Persist external data to JSON cache."""
    serialisable = {str(k): asdict(v) for k, v in data.items()}
    _CACHE_PATH.write_text(json.dumps(serialisable, indent=2))
    logger.info("External data cached to %s", _CACHE_PATH)


def _load_cache() -> Optional[dict[int, ExternalCouncilData]]:
    """Load external data from JSON cache if available."""
    if not _CACHE_PATH.exists():
        return None

    try:
        raw = json.loads(_CACHE_PATH.read_text())
        result: dict[int, ExternalCouncilData] = {}
        for cid_str, vals in raw.items():
            result[int(cid_str)] = ExternalCouncilData(
                hdt_measurement=vals.get("hdt_measurement"),
                has_green_belt=vals.get("has_green_belt", False),
            )
        logger.info("Loaded external data from cache (%d councils)", len(result))
        return result
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to load external data cache: %s", exc)
        return None


def load_external_data(*, force_rebuild: bool = False) -> dict[int, ExternalCouncilData]:
    """Load external council data, using cache when available.

    Args:
        force_rebuild: If True, ignore cache and re-parse source files.

    Returns:
        Mapping of council_id → ExternalCouncilData.
    """
    if not force_rebuild:
        cached = _load_cache()
        if cached is not None:
            return cached

    data = _build_external_data()
    _save_cache(data)
    return data
