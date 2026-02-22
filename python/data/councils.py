"""Canonical council registry, loaded from council_list.xlsx.

Provides a ``resolve(name)`` function that maps any reasonable variant of a
council name to its canonical :class:`Council` entry.  If a name cannot be
matched it returns ``None`` — callers should discard that row.

Resolution pipeline (first match wins):
  1. Exact match against canonical names.
  2. Explicit alias lookup (hand-crafted variants, see ``_ALIASES``).
  3. Automated normalisation (strip common prefixes/suffixes, lower-case,
     replace `` & `` → `` and ``), then lookup.
  4. Case-insensitive exact match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Council:
    id: int
    name: str


# ---------------------------------------------------------------------------
# Load registry
# ---------------------------------------------------------------------------


def _load() -> dict[int, Council]:
    path = Path(__file__).resolve().parent / "council_list.xlsx"
    df = pd.read_excel(path)
    return {
        int(row["id"]): Council(id=int(row["id"]), name=str(row["name"]).strip())
        for _, row in df.iterrows()
    }


#: Full registry: id → Council
COUNCILS: dict[int, Council] = _load()

#: Reverse lookup: canonical name → id
_BY_NAME: dict[str, int] = {c.name: c.id for c in COUNCILS.values()}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = (
    "London Borough of ",
    "Royal Borough of ",
    "City of ",
    "Borough of ",
    "County of ",
)

_STRIP_SUFFIXES = (
    " Development Corporation",
    " City Council",
    " Borough Council",
    " District Council",
    " County Council",
    " Corporation",
    " Council",
)


def _normalise(name: str) -> str:
    """Strip common prefixes/suffixes, replace & with and, lower-case."""
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace(" & ", " and ").strip().lower()


#: Normalised lookup: normalised(canonical name) → id
_BY_NORM: dict[str, int] = {_normalise(c.name): c.id for c in COUNCILS.values()}


# ---------------------------------------------------------------------------
# Explicit aliases
# Covers cases the automated pipeline cannot handle: renamed councils,
# mergers, abbreviations, typos, and punctuation/capitalisation variants.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # --- Ghost permit abbreviations / prefixes ---
    "LLDC": "London Legacy Development",
    "Richmond": "Richmond upon Thames",
    "Royal Borough of Kingston (LA Code)": "Kingston",
    "Kingston upon Thames": "Kingston",
    "Greenwich": "Royal Greenwich",

    # --- Corporate suffix variants (not covered by strip pipeline) ---
    "London Legacy Development Corporation": "London Legacy Development",
    "Old Oak and Park Royal Development Corporation": "Old Oak and Park Royal",

    # --- City suffix — data uses short form, registry uses "X City" ---
    "Birmingham": "Birmingham City",
    "Canterbury": "Canterbury City",
    "Coventry": "Coventry City",
    "Gloucester": "Gloucester City",
    "Leeds": "Leeds City",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool City",
    "Manchester": "Manchester City",
    "Nottingham": "Nottingham City",
    "Sheffield": "Sheffield City",
    "St Albans": "St Albans City",
    "Worcester": "Worcester City",

    # --- "X, City of" / "X, County of" inversions ---
    "Bristol, City of": "Bristol",
    "Herefordshire, County of": "Herefordshire",
    "Kingston upon Hull, City of": "Hull City",

    # --- Hull ---
    "Kingston upon Hull": "Hull City",

    # --- County / administrative prefix variants ---
    "County Durham": "Durham",
    "Conwy": "Conwy County",

    # --- Punctuation / spacing variants ---
    "Epsom and Ewell": "Epsom Ewell",
    "St. Helens": "St Helens",
    "St. Helels": "St Helens",           # typo present in income data
    "Castle Point": "Castlepoint",
    "Southend-on-Sea": "Southend on Sea",
    "Milton Keynes": "Milton-Keynes",

    # --- Bournemouth merger ---
    "Bournemouth": "Bournemouth, Christchurch, Poole",
    "Bournemouth, Christchurch and Poole": "Bournemouth, Christchurch, Poole",

    # --- King's Lynn apostrophe variant ---
    "King's Lynn and West Norfolk": "Kings Lynn & West Norfolk",

    # --- Merged / restructured councils ---
    # Adur + Worthing → combined entry
    "Adur": "Adur and Worthing",
    "Worthing": "Adur and Worthing",
    # Babergh + Mid Suffolk → combined entry
    "Babergh": "Babergh and Mid-Suffolk",
    "Mid Suffolk": "Babergh and Mid-Suffolk",
    # Cambridge + South Cambridgeshire → combined entry
    "Cambridge": "Cambridge City and South Cambridgeshire",
    "South Cambridgeshire": "Cambridge City and South Cambridgeshire",
    # Broadland + South Norfolk → combined entry
    "Broadland": "South Norfolk and Broadland",
    "South Norfolk": "South Norfolk and Broadland",
    # Maidstone + Swale → combined entry
    "Maidstone": "Maidstone and Swale",
    "Swale": "Maidstone and Swale",
    # Bromsgrove + Redditch → combined entry
    "Bromsgrove": "Bromsgrove and Redditch",
    "Redditch": "Bromsgrove and Redditch",
    # Central + South Bedfordshire → combined entry
    "Central Bedfordshire": "Central & South Bedfordshire",
    # Eden + South Lakeland → Westmorland & Furness (2023 merger)
    "Eden": "Westmorland & Furness",
    "South Lakeland": "Westmorland & Furness",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(name: str) -> Optional[Council]:
    """Resolve *name* to a canonical :class:`Council`, or ``None`` if unmatched.

    Args:
        name: Any council name variant from a data source.

    Returns:
        A :class:`Council` if matched, ``None`` otherwise.
    """
    if not name or not isinstance(name, str):
        return None

    name = name.strip()

    # 1. Exact match
    if name in _BY_NAME:
        return COUNCILS[_BY_NAME[name]]

    # 2. Explicit alias
    canonical = _ALIASES.get(name)
    if canonical and canonical in _BY_NAME:
        return COUNCILS[_BY_NAME[canonical]]

    # 3. Normalise, then lookup (handles prefix stripping, & → and, lower-case)
    norm = _normalise(name)
    if norm in _BY_NORM:
        return COUNCILS[_BY_NORM[norm]]

    # 3b. Normalise alias target too (e.g. "London Borough of Hammersmith and Fulham")
    stripped = name
    for prefix in _STRIP_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped != name:
        # Try exact and alias on the stripped form
        if stripped in _BY_NAME:
            return COUNCILS[_BY_NAME[stripped]]
        canonical = _ALIASES.get(stripped)
        if canonical and canonical in _BY_NAME:
            return COUNCILS[_BY_NAME[canonical]]
        # Normalise the stripped form
        norm_stripped = _normalise(stripped)
        if norm_stripped in _BY_NORM:
            return COUNCILS[_BY_NORM[norm_stripped]]

    # 4. Case-insensitive fallback
    lower = name.lower()
    for canon, cid in _BY_NAME.items():
        if canon.lower() == lower:
            return COUNCILS[cid]

    return None
