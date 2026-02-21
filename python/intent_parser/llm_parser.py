"""LLM-based intent parser using Google Gemini.

Takes a raw natural language query and returns a ``ParsedIntent`` by prompting
Gemini with structured output instructions, then enriching the result with
location data.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date, timedelta
from typing import Any

from .schema import (
    AnalysisGoal,
    Constraint,
    DevelopmentIntent,
    LocationIntent,
    ParsedIntent,
)
from . import location as loc


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntentParseError(Exception):
    """Raised when the LLM call or response parsing fails."""


# ---------------------------------------------------------------------------
# Radius defaults by location level
# ---------------------------------------------------------------------------

_RADIUS_BY_LEVEL: dict[str, int] = {
    "address": 200,
    "neighbourhood": 800,
    "borough": 3_000,
    "city": 5_000,
    "region": 15_000,
    "county": 20_000,
    "country": 50_000,
    "unspecified": 5_000,
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a UK planning intelligence parser. Your job is to take a natural \
language query about property development or planning in the UK and extract \
structured data from it.

## UK Planning Domain Knowledge

Use classes:
- C3 = dwelling houses (residential)
- C4 = houses in multiple occupation (HMO, up to 6 people)
- E = commercial, business and service (shops, offices, restaurants, gyms, clinics, creches)
- F1 = learning and non-residential institutions (schools, museums, libraries, churches)
- F2 = local community (small shops, halls, outdoor sport)
- B2 = general industrial
- B8 = storage and distribution
- sui generis = uses that don't fall into any class (theatres, pubs, hot food takeaways, petrol stations, nightclubs)

Scale definitions (England):
- minor: fewer than 10 dwellings, or under 1,000 sqm floor space
- major: 10–99 dwellings, or 1,000–9,999 sqm floor space
- large major: 100+ dwellings, or 10,000+ sqm floor space

Common London neighbourhood → borough mappings:
- Shoreditch, Dalston, Hackney Wick → Hackney
- Brixton, Clapham, Streatham → Lambeth
- Peckham, Camberwell, Bermondsey → Southwark
- Soho, Mayfair, Paddington → Westminster
- Canary Wharf, Bow, Whitechapel → Tower Hamlets
- Camden Town, Hampstead, King's Cross → Camden
- Angel, Highbury, Holloway → Islington
- Wimbledon, Mitcham → Merton
- Tottenham, Wood Green → Haringey

## Output Schema

You MUST output ONLY valid JSON matching this schema (no markdown fences, no \
commentary, no extra keys):

{schema}

## Examples

INPUT: "I want to build 20 affordable flats in South London, avoiding flood zones"
OUTPUT:
{{"development":{{"category":"residential","subcategory":"affordable_housing","description":"20-unit affordable housing block","scale":"major","unit_count":20,"use_class":"C3","from_use":null,"to_use":null,"raw_tags":["affordable","flats"]}},"location":{{"raw_text":"South London","level":"region","names":["South London"],"resolved_councils":[],"resolved_coordinates":null,"radius_suggestion_m":null,"country":"England"}},"constraints":[{{"type":"avoid","category":"flood_risk","value":null,"raw_text":"avoiding flood zones"}}],"analysis_goals":[{{"goal":"find_sites","detail":null,"time_range":null}}],"keywords":["affordable","flats","South London"],"confidence":0.92,"ambiguities":["'affordable' could mean social housing (below 80% market rent) or shared ownership"]}}

INPUT: "Convert an old warehouse in Shoreditch into a boutique hotel"
OUTPUT:
{{"development":{{"category":"change_of_use","subcategory":"warehouse_conversion","description":"Warehouse to boutique hotel conversion","scale":"major","unit_count":null,"use_class":"sui_generis","from_use":"B8","to_use":"C1","raw_tags":["boutique","hotel","warehouse","conversion"]}},"location":{{"raw_text":"Shoreditch","level":"neighbourhood","names":["Shoreditch"],"resolved_councils":[],"resolved_coordinates":null,"radius_suggestion_m":null,"country":"England"}},"constraints":[],"analysis_goals":[{{"goal":"check_feasibility","detail":"likelihood of planning approval for warehouse-to-hotel conversion in Shoreditch","time_range":null}}],"keywords":["warehouse","conversion","hotel","Shoreditch"],"confidence":0.88,"ambiguities":["Number of hotel rooms not specified","'Old warehouse' — listed building status unknown"]}}

INPUT: "Which boroughs in London have the highest approval rate for HMO conversions?"
OUTPUT:
{{"development":{{"category":"change_of_use","subcategory":"HMO","description":"HMO conversion","scale":"minor","unit_count":null,"use_class":"C4","from_use":"C3","to_use":"C4","raw_tags":["HMO","conversion"]}},"location":{{"raw_text":"London","level":"region","names":["London"],"resolved_councils":[],"resolved_coordinates":null,"radius_suggestion_m":null,"country":"England"}},"constraints":[],"analysis_goals":[{{"goal":"compare_areas","detail":"which London boroughs have the highest approval rate for HMO conversions","time_range":null}}],"keywords":["HMO","conversion","approval rate"],"confidence":0.95,"ambiguities":[]}}

## Critical Instructions

- Populate `ambiguities` with anything unclear or assumed. Be honest.
- Set `confidence` below 0.7 if the query is vague or missing key information.
- If the query is clearly not about property development or planning, set \
confidence to 0.0 and add "Query does not appear to be about property \
development or planning" to ambiguities.
- Do NOT include the `id` or `raw_query` fields — those are added automatically.
- Output ONLY valid JSON. No markdown fences. No explanation.
"""


def _build_system_prompt() -> str:
    """Build the system prompt, embedding the live JSON schema."""
    schema = ParsedIntent.model_json_schema()
    # Remove fields the LLM shouldn't set.
    props = schema.get("properties", {})
    props.pop("id", None)
    props.pop("raw_query", None)
    schema_str = json.dumps(schema, indent=2)
    return _SYSTEM_PROMPT_TEMPLATE.format(schema=schema_str)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating markdown fences."""
    text = text.strip()
    # Try raw parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fences.
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise IntentParseError(f"Could not parse JSON from LLM response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def _enrich(intent: ParsedIntent) -> ParsedIntent:
    """Enrich a ParsedIntent with location resolution and sensible defaults."""
    # Location enrichment via our geography module.
    location_text = intent.location.raw_text
    if location_text and location_text.lower() != "unspecified":
        result = loc.resolve(location_text)
        if result["councils"]:
            intent.location.resolved_councils = result["councils"]
        if result["coordinates"]:
            intent.location.resolved_coordinates = result["coordinates"]
        # Use resolved level if ours is unspecified.
        if intent.location.level == "unspecified" and result["level"] != "unspecified":
            intent.location.level = result["level"]

    # Infer radius from level if not already set.
    if intent.location.radius_suggestion_m is None:
        intent.location.radius_suggestion_m = _RADIUS_BY_LEVEL.get(
            intent.location.level, 5_000
        )

    # Default time_range to last 3 years for find_sites goals.
    today = date.today()
    three_years_ago = (today - timedelta(days=3 * 365)).isoformat()
    today_str = today.isoformat()
    for goal in intent.analysis_goals:
        if goal.goal == "find_sites" and goal.time_range is None:
            goal.time_range = {"from": three_years_ago, "to": today_str}

    return intent


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def _build_intent_from_llm_data(query: str, data: dict[str, Any]) -> ParsedIntent:
    """Construct and enrich a ParsedIntent from LLM JSON output."""
    # Strip fields the LLM shouldn't control.
    data.pop("id", None)
    data.pop("raw_query", None)

    intent = ParsedIntent(raw_query=query, **data)
    return _enrich(intent)


async def parse_query(query: str, api_key: str | None = None) -> ParsedIntent:
    """Parse a natural language planning query into a structured ParsedIntent.

    Uses Google Gemini (gemini-2.0-flash by default) to interpret the query,
    then enriches the result with UK location data.

    Args:
        query: The raw user input, e.g. "I want to build 20 affordable flats
               in South London, avoiding flood zones".
        api_key: Google AI API key.  Falls back to the ``GOOGLE_API_KEY``
                 environment variable if not provided.

    Returns:
        A fully populated and enriched ``ParsedIntent``.

    Raises:
        IntentParseError: If the Gemini call fails or the response can't be
            parsed into valid JSON / a valid ``ParsedIntent``.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise IntentParseError(
            "No API key provided. Pass api_key= or set GOOGLE_API_KEY."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise IntentParseError(
            "google-genai package is required: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=key)
    system_prompt = _build_system_prompt()

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=2000,
            ),
        )
    except Exception as exc:
        raise IntentParseError(f"Gemini API call failed: {exc}") from exc

    raw_text = response.text or ""

    # First attempt at JSON extraction.
    try:
        data = _extract_json(raw_text)
    except IntentParseError:
        # Retry with a stricter nudge.
        try:
            retry_response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    f"Your previous response was not valid JSON. "
                    f"Parse this planning query and respond with ONLY valid "
                    f"JSON, no markdown fences:\n\n{query}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    max_output_tokens=2000,
                ),
            )
            data = _extract_json(retry_response.text or "")
        except Exception as retry_exc:
            raise IntentParseError(
                f"Failed to parse JSON after retry. "
                f"Original response:\n{raw_text[:500]}"
            ) from retry_exc

    try:
        return _build_intent_from_llm_data(query, data)
    except Exception as exc:
        raise IntentParseError(
            f"Failed to construct ParsedIntent from LLM output: {exc}\n"
            f"Data: {json.dumps(data, indent=2)[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def parse_query_sync(query: str, api_key: str | None = None) -> ParsedIntent:
    """Synchronous wrapper for :func:`parse_query`."""
    return asyncio.run(parse_query(query, api_key=api_key))


# ---------------------------------------------------------------------------
# Mock parser for testing without an API key
# ---------------------------------------------------------------------------


def parse_query_mock(query: str) -> ParsedIntent:
    """Return a plausible ``ParsedIntent`` for testing — no API key needed.

    Picks a response template based on keywords found in the query.
    The result is still enriched through the location resolver.
    """
    q = query.lower()

    # ----- HMO / conversion ------------------------------------------------
    if "hmo" in q or "house in multiple" in q:
        data: dict[str, Any] = {
            "development": {
                "category": "change_of_use",
                "subcategory": "HMO",
                "description": "HMO conversion",
                "scale": "minor",
                "use_class": "C4",
                "from_use": "C3",
                "to_use": "C4",
                "raw_tags": ["HMO", "conversion"],
            },
            "location": _mock_location(q),
            "constraints": [],
            "analysis_goals": [{"goal": "check_feasibility"}],
            "keywords": ["HMO", "conversion"],
            "confidence": 0.85,
            "ambiguities": ["Number of bedrooms not specified"],
        }

    # ----- Hotel / hospitality ----------------------------------------------
    elif "hotel" in q or "hospitality" in q:
        data = {
            "development": {
                "category": "hospitality",
                "subcategory": "hotel",
                "description": "Hotel development",
                "scale": "major",
                "use_class": "sui_generis",
                "raw_tags": ["hotel"],
            },
            "location": _mock_location(q),
            "constraints": [],
            "analysis_goals": [{"goal": "check_feasibility"}],
            "keywords": ["hotel"],
            "confidence": 0.80,
            "ambiguities": ["Number of rooms not specified", "Hotel class unknown"],
        }

    # ----- Extension / home improvement -------------------------------------
    elif any(w in q for w in ("extension", "loft", "conservatory", "garage")):
        data = {
            "development": {
                "category": "home_improvement",
                "subcategory": "rear_extension" if "rear" in q else "loft_extension" if "loft" in q else "extension",
                "description": "Residential extension",
                "scale": "minor",
                "use_class": "C3",
                "raw_tags": ["extension"],
            },
            "location": _mock_location(q),
            "constraints": [],
            "analysis_goals": [{"goal": "check_feasibility"}],
            "keywords": ["extension", "home improvement"],
            "confidence": 0.82,
            "ambiguities": ["Exact size not specified"],
        }

    # ----- Commercial -------------------------------------------------------
    elif any(w in q for w in ("office", "commercial", "shop", "retail")):
        data = {
            "development": {
                "category": "commercial",
                "description": "Commercial development",
                "scale": "major",
                "use_class": "E",
                "raw_tags": ["commercial"],
            },
            "location": _mock_location(q),
            "constraints": [],
            "analysis_goals": [{"goal": "find_sites"}],
            "keywords": ["commercial"],
            "confidence": 0.78,
            "ambiguities": ["Specific commercial use not clear"],
        }

    # ----- Default: residential ---------------------------------------------
    else:
        # Try to extract a unit count.
        import re as _re
        unit_match = _re.search(r"(\d+)\s+(?:\w+\s+)*?(?:units?|flats?|homes?|dwellings?|houses?)", q)
        unit_count = int(unit_match.group(1)) if unit_match else None
        scale = None
        if unit_count is not None:
            scale = "minor" if unit_count < 10 else ("large_major" if unit_count >= 100 else "major")

        tags = []
        for tag in ("affordable", "social", "luxury", "eco", "modular", "flats", "houses"):
            if tag in q:
                tags.append(tag)

        constraints = []
        if "flood" in q:
            constraints.append({
                "type": "avoid",
                "category": "flood_risk",
                "raw_text": "flood risk",
            })
        if "conservation" in q:
            constraints.append({
                "type": "avoid",
                "category": "conservation_area",
                "raw_text": "conservation area",
            })
        if "green belt" in q:
            constraints.append({
                "type": "avoid",
                "category": "green_belt",
                "raw_text": "green belt",
            })

        desc_parts = []
        if unit_count:
            desc_parts.append(f"{unit_count}-unit")
        if "affordable" in q:
            desc_parts.append("affordable")
        desc_parts.append("residential development")
        description = " ".join(desc_parts)

        data = {
            "development": {
                "category": "residential",
                "subcategory": "affordable_housing" if "affordable" in q else None,
                "description": description,
                "scale": scale,
                "unit_count": unit_count,
                "use_class": "C3",
                "raw_tags": tags or ["residential"],
            },
            "location": _mock_location(q),
            "constraints": constraints,
            "analysis_goals": [{"goal": "find_sites"}],
            "keywords": tags or ["residential"],
            "confidence": 0.85,
            "ambiguities": [],
        }

    return _build_intent_from_llm_data(query, data)


def _mock_location(query_lower: str) -> dict[str, Any]:
    """Extract a rough location dict from a lowercased query for mocking."""
    # Check known locations.
    known = [
        ("south london", "region"),
        ("north london", "region"),
        ("east london", "region"),
        ("west london", "region"),
        ("central london", "region"),
        ("greater manchester", "region"),
        ("london", "region"),
    ]
    for name, level in known:
        if name in query_lower:
            return {
                "raw_text": name.title(),
                "level": level,
                "names": [name.title()],
                "country": "England",
            }

    # Check council names from our location module.
    for council_name in loc.UK_COUNCILS:
        if council_name.lower() in query_lower:
            data = loc.UK_COUNCILS[council_name]
            is_london = data["region"] == "London"
            return {
                "raw_text": council_name,
                "level": "borough" if is_london else "city",
                "names": [council_name],
                "country": data["country"],
            }

    # Check aliases.
    for alias_lower, council_name in loc._ALIAS_INDEX.items():
        if alias_lower in query_lower:
            return {
                "raw_text": alias_lower.title(),
                "level": "neighbourhood",
                "names": [alias_lower.title()],
                "country": loc.UK_COUNCILS[council_name]["country"],
            }

    return {
        "raw_text": "unspecified",
        "level": "unspecified",
        "names": [],
        "country": "England",
    }
