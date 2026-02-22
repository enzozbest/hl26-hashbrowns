"""NLU: parse a user's free-text proposal into structured intent.

Uses rule-based extraction with regex patterns to pull structured fields
from a natural-language description of a planning proposal.  Falls back to
heuristics when regex patterns do not match.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


# ── Output model ─────────────────────────────────────────────────────────────


class ProposalIntent(BaseModel):
    """Structured representation of a parsed planning proposal."""

    raw_text: str = Field(..., description="Original free-text description")
    num_houses: Optional[int] = Field(
        None, description="Extracted number of dwellings",
    )
    project_type: str = Field(
        "mixed",
        description=(
            "Inferred category: small residential / medium residential / "
            "large residential / home improvement / mixed"
        ),
    )
    region: Optional[str] = Field(
        None,
        description="UK region (e.g. South England, Greater London)",
    )
    specific_council: Optional[str] = Field(
        None, description="Specific council name if mentioned",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Extracted relevant terms",
    )
    unit_mix_preference: Optional[dict[str, int]] = Field(
        None,
        description="Unit-mix breakdown if specified (e.g. {'three_bed': 5})",
    )


# ── Constants ────────────────────────────────────────────────────────────────

# UK regions → example council keywords (for matching).
_UK_REGIONS: dict[str, list[str]] = {
    "Greater London": [
        "london", "westminster", "camden", "islington", "hackney",
        "tower hamlets", "southwark", "lambeth", "lewisham", "greenwich",
        "barnet", "brent", "ealing", "haringey", "newham", "croydon",
        "bromley", "enfield", "hillingdon", "hounslow", "wandsworth",
        "richmond", "kingston", "merton", "sutton", "havering",
        "redbridge", "waltham forest", "harrow", "kensington", "chelsea",
        "hammersmith", "fulham", "barking", "dagenham", "bexley",
    ],
    "South East": [
        "surrey", "kent", "sussex", "hampshire", "berkshire",
        "buckinghamshire", "oxfordshire", "brighton", "portsmouth",
        "southampton", "milton keynes", "reading", "slough", "windsor",
        "maidenhead", "guildford", "canterbury", "dover", "crawley",
        "hastings", "isle of wight", "medway", "woking",
    ],
    "South West": [
        "bristol", "bath", "devon", "cornwall", "dorset", "somerset",
        "wiltshire", "gloucester", "plymouth", "exeter", "bournemouth",
        "swindon", "cheltenham", "torbay", "poole", "taunton",
    ],
    "South England": [
        "south england", "south of england", "southern england",
    ],
    "East of England": [
        "essex", "hertfordshire", "cambridgeshire", "norfolk", "suffolk",
        "bedfordshire", "luton", "peterborough", "ipswich", "norwich",
        "colchester", "chelmsford", "southend", "thurrock", "basildon",
    ],
    "Midlands": [
        "midlands", "birmingham", "coventry", "wolverhampton", "derby",
        "leicester", "nottingham", "stoke", "warwickshire", "staffordshire",
        "shropshire", "worcestershire", "herefordshire", "northampton",
        "lincoln", "telford",
    ],
    "North West": [
        "manchester", "liverpool", "cheshire", "lancashire", "cumbria",
        "merseyside", "bolton", "wigan", "salford", "stockport",
        "oldham", "blackburn", "blackpool", "preston", "warrington",
        "chester", "carlisle",
    ],
    "North East": [
        "newcastle", "sunderland", "durham", "northumberland",
        "middlesbrough", "hartlepool", "darlington", "gateshead",
        "south tyneside", "north tyneside", "stockton",
    ],
    "Yorkshire": [
        "yorkshire", "leeds", "sheffield", "bradford", "york",
        "huddersfield", "hull", "doncaster", "barnsley", "rotherham",
        "wakefield", "halifax", "harrogate", "scarborough",
    ],
    "Wales": [
        "wales", "cardiff", "swansea", "newport", "wrexham", "bangor",
        "aberystwyth", "carmarthen", "pembroke", "ceredigion", "gwynedd",
        "powys", "bridgend", "neath", "port talbot", "merthyr",
        "caerphilly", "rhondda", "cynon", "taf", "vale of glamorgan",
    ],
    "Scotland": [
        "scotland", "edinburgh", "glasgow", "aberdeen", "dundee",
        "inverness", "stirling", "perth", "fife", "highland",
        "falkirk", "ayrshire", "lothian", "borders",
    ],
}

# Words that carry domain signal when found in a proposal.
_SIGNAL_WORDS: set[str] = {
    "residential", "commercial", "industrial", "retail", "office",
    "warehouse", "agricultural", "mixed-use", "mixed use",
    "affordable", "social", "sheltered", "student", "hotel",
    "extension", "conversion", "demolition", "refurbishment",
    "renovation", "change of use", "outline", "reserved matters",
    "full application", "listed building", "conservation",
    "green belt", "flood zone", "brownfield", "greenfield",
    "sustainable", "eco", "parking", "access", "landscaping",
    "detached", "semi-detached", "terraced", "bungalow",
    "apartment", "flat", "flats", "maisonette", "townhouse",
    "storey", "stories", "storeys", "basement",
    "garage", "annex", "loft", "dormer", "outbuilding",
}

# English stop words (compact set for keyword filtering).
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "i", "we", "you",
    "he", "she", "it", "they", "me", "us", "him", "her", "them", "my",
    "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very", "just",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "also", "new", "want", "like", "need",
    "build", "propose", "plan", "planning", "application", "proposal",
    "proposed", "erection", "construction", "development", "site", "land",
}

# Named number words.
_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "a single": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100,
}

# Bedroom-type patterns → canonical key.
_BED_PATTERNS: dict[str, str] = {
    r"(?:1|one)[\s-]*bed": "one_bed",
    r"(?:2|two)[\s-]*bed": "two_bed",
    r"(?:3|three)[\s-]*bed": "three_bed",
    r"(?:4|four)[\s\+-]*bed": "four_plus_bed",
    r"(?:5|five)[\s\+-]*bed": "four_plus_bed",
    r"studio": "one_bed",
}


# ── Parser ───────────────────────────────────────────────────────────────────


class ProposalParser:
    """Parse natural-language planning proposals into structured intent.

    All extraction is rule-based (regex + heuristics).  No external model
    or API calls are made.
    """

    def parse(self, proposal: str) -> ProposalIntent:
        """Parse a free-text proposal description.

        Args:
            proposal: Raw user input.

        Returns:
            A :class:`ProposalIntent` with extracted fields populated.
        """
        text = proposal.strip()
        lower = text.lower()

        num_houses = self._extract_num_houses(lower)
        unit_mix = self._extract_unit_mix(lower)
        project_type = self._infer_project_type(num_houses, lower)
        region = self._extract_region(lower)
        council = self._extract_council(lower)
        keywords = self._extract_keywords(lower, num_houses)

        return ProposalIntent(
            raw_text=text,
            num_houses=num_houses,
            project_type=project_type,
            region=region,
            specific_council=council,
            keywords=keywords,
            unit_mix_preference=unit_mix if unit_mix else None,
        )

    # ── extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_num_houses(text: str) -> Optional[int]:
        """Extract the number of proposed dwellings.

        Handles numeric patterns ("200 houses") and word-based patterns
        ("a single house").
        """
        dwelling_words = (
            r"(?:houses?|homes?|dwellings?|units?|"
            r"flats?|apartments?|bungalows?|"
            r"maisonettes?|townhouses?|properties|residences?)"
        )

        # "200 houses", "10 new homes", "3 residential units",
        # "100 affordable flats", "50 detached houses"
        m = re.search(
            rf"(\d[\d,]*)\s+(?:\w+\s+){{0,2}}{dwelling_words}",
            text,
        )
        if m:
            return int(m.group(1).replace(",", ""))

        # "a single house", "one dwelling", "two detached houses"
        # Allow up to two optional adjectives between the number and dwelling.
        adj_gap = r"(?:\w+\s+){0,2}"
        for word, val in sorted(
            _NUMBER_WORDS.items(), key=lambda kv: -len(kv[0]),
        ):
            pattern = rf"\b{re.escape(word)}\s+{adj_gap}{dwelling_words}"
            if re.search(pattern, text):
                return val

        return None

    @staticmethod
    def _extract_unit_mix(text: str) -> dict[str, int]:
        """Extract bedroom-type breakdown (e.g. '3 x 3-bed houses')."""
        mix: dict[str, int] = {}

        for pattern, key in _BED_PATTERNS.items():
            # "3 x 3-bed" or "three 3-bed" or "10 three-bed houses"
            m = re.search(
                rf"(\d+)\s*(?:x\s*)?{pattern}", text,
            )
            if m:
                mix[key] = int(m.group(1))

        return mix

    @staticmethod
    def _infer_project_type(num_houses: Optional[int], text: str) -> str:
        """Infer project type from house count and keywords."""
        # Home improvement signals (do NOT include "change of use" —
        # that is typically a commercial / mixed conversion, not a home
        # improvement).
        home_improvement_patterns = [
            r"\bextension\b", r"loft\s+conversion", r"\bdormer\b",
            r"\bconservatory\b", r"garage\s+conversion", r"\bannex\b",
            r"\boutbuilding\b", r"\bporch\b",
            r"\brenovation\b", r"\brefurbishment\b",
            r"internal\s+alterations?",
            r"single\s+storey\s+rear", r"two\s+storey\s+(?:side|rear)",
        ]
        for p in home_improvement_patterns:
            if re.search(p, text):
                return "home improvement"

        # Commercial / industrial signals override residential inference
        if re.search(r"\b(?:office|retail|warehouse|industrial|commercial)\b", text):
            if num_houses is None or num_houses == 0:
                return "mixed"

        if num_houses is not None:
            if num_houses <= 2:
                return "small residential"
            if num_houses <= 24:
                return "medium residential"
            return "large residential"

        # Residential keywords without a count
        if re.search(
            r"\b(?:house|home|dwelling|flat|apartment|bungalow|residential)\b",
            text,
        ):
            return "small residential"

        return "mixed"

    @staticmethod
    def _extract_region(text: str) -> Optional[str]:
        """Match text against UK regions."""
        # Check for direct region name mentions first
        for region in _UK_REGIONS:
            if region.lower() in text:
                return region

        # Check council/area keyword matches
        for region, keywords in _UK_REGIONS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    return region

        return None

    @staticmethod
    def _extract_council(text: str) -> Optional[str]:
        """Extract a specific council name if mentioned.

        Looks for patterns like "X Borough Council", "X City Council",
        "X District Council", "X County Council", or "council of X".
        """
        # Match council names. The name group uses a tight pattern to
        # avoid capturing preceding verbs/prepositions.
        patterns = [
            # "London Borough of Hackney" → "Hackney"
            r"london\s+borough\s+of\s+([a-z]+)",
            # "Westminster City Council" — capture the single proper noun
            # immediately before the council-type keyword.
            r"\b([a-z]+)\s+(?:borough|city|district|county|metropolitan)\s+council",
            # "council of X" / "council of the X" → X
            r"council\s+of\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+){0,3}?)(?:\s*[,.]|\s*$)",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                name = m.group(1).strip().title()
                if len(name) > 2 and name.lower() not in _STOP_WORDS:
                    return name
        return None

    @staticmethod
    def _extract_keywords(text: str, num_houses: Optional[int]) -> list[str]:
        """Extract domain-relevant keywords from the text."""
        # Find words/phrases that are signal words
        found: list[str] = []
        for phrase in sorted(_SIGNAL_WORDS, key=len, reverse=True):
            if phrase in text:
                found.append(phrase)
                # Remove to avoid sub-phrase duplication
                text = text.replace(phrase, " ")

        # Also extract remaining content words (nouns/adjectives heuristic:
        # multi-character words not in stop words or already captured).
        tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", text)
        for tok in tokens:
            if (
                tok not in _STOP_WORDS
                and len(tok) > 2
                and tok not in found
                and not tok.isdigit()
            ):
                found.append(tok)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for kw in found:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)

        return deduped
