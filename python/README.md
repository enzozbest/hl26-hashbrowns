# Planning Site Finder — Intent Parser

NLP-powered intent parser that converts natural language planning queries into structured, API-agnostic domain models. These models describe *what the user wants* without coupling to any specific data provider.

**Example input:**

> "I want to build 20 affordable flats in South London, avoiding flood zones"

**Parsed output:** a `ParsedIntent` capturing development details, location, constraints, analysis goals, and extracted keywords — ready to be translated into any downstream API by an adapter.

## Project Structure

```
intent_parser/
├── __init__.py          # Package exports
├── schema.py            # Canonical domain models (Pydantic v2)
├── llm_parser.py        # LLM-based query → ParsedIntent (scaffold)
├── location.py          # Location enrichment / geocoding (scaffold)
└── adapters/
    ├── __init__.py
    └── base.py          # Abstract BaseAdapter interface
```

## Schema Overview

### Models

| Model | Description |
|---|---|
| `ParsedIntent` | Top-level container — UUID, raw query, all parsed fields, confidence score, ambiguities |
| `DevelopmentIntent` | What to build — category, subcategory, scale, unit count, use classes, change-of-use support |
| `LocationIntent` | Where — raw text, granularity level, names, with enrichment fields for councils & coordinates |
| `Constraint` | Flexible avoid / require / prefer with freeform category and value |
| `AnalysisGoal` | What the user wants to learn — find sites, check feasibility, compare areas, etc. |

### Key Design Principles

- **Zero API coupling** — the schema describes user intent in pure domain terms. No IBex, no EPC register, nothing external.
- **Enrichment-ready** — `LocationIntent.resolved_councils` and `resolved_coordinates` start empty and get populated by the location enrichment step.
- **Flexible constraints** — freeform `category` / `value` pairs instead of hardcoded booleans, so new constraint types don't require schema changes.
- **Adapter pattern** — each external API gets a `BaseAdapter` subclass that translates `ParsedIntent` into its own request format.

## Usage

```python
from intent_parser.schema import (
    ParsedIntent, DevelopmentIntent, LocationIntent,
    Constraint, AnalysisGoal,
)

intent = ParsedIntent(
    raw_query="I want to build 20 affordable flats in South London, avoiding flood zones",
    development=DevelopmentIntent(
        category="residential",
        subcategory="affordable_housing",
        description="20-unit affordable housing block",
        scale="major",
        unit_count=20,
        use_class="C3",
        raw_tags=["affordable", "flats"],
    ),
    location=LocationIntent(
        raw_text="South London",
        level="region",
        names=["South London"],
    ),
    constraints=[
        Constraint(type="avoid", category="flood_risk", raw_text="avoiding flood zones"),
    ],
    analysis_goals=[
        AnalysisGoal(goal="find_sites"),
    ],
    keywords=["affordable", "flats", "South London"],
    confidence=0.9,
    ambiguities=['"affordable" could mean social housing or below-market-rate'],
)
```

### Serialisation

```python
# Full JSON-compatible dict
data = intent.to_dict()

# Human-readable summary
print(intent.to_summary())
# 20-unit affordable housing block in South London.
# Constraints: avoid flood_risk.
# Goals: find sites.
```

### Writing an Adapter

Subclass `BaseAdapter` to translate a `ParsedIntent` into a specific API's request format:

```python
from intent_parser.adapters.base import BaseAdapter
from intent_parser.schema import ParsedIntent

class MyAPIAdapter(BaseAdapter):
    def build_payloads(self, intent: ParsedIntent) -> list[dict]:
        # Convert intent into API-specific request bodies
        ...

    async def execute(self, intent: ParsedIntent) -> list[dict]:
        payloads = self.build_payloads(intent)
        # Call the API, return normalised results
        ...
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.14+.
