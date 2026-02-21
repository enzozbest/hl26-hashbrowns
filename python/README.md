# Planning Site Finder

NLP-powered planning intelligence tool. Type a natural language query like:

> "I want to build 20 affordable flats in South London, avoiding flood zones"

and get structured search results from planning data APIs.

## How It Works

```
Frontend (Next.js)
    ↓  user types natural language query
    ↓
POST /api/parse
    ↓
LLM Parser (Gemini)         → ParsedIntent (structured JSON)
    ↓
Location Resolver            → Enriches with council names, coordinates
    ↓
POST /api/search
    ↓
Search Orchestrator          → Fans out to all adapters concurrently
    ├── IBex Adapter         → Planning applications, approval/refusal data
    ├── EPC Adapter          → Energy performance ratings
    ├── Flood Adapter        → Flood zone checks
    ├── Price Paid Adapter   → Land Registry transaction history
    └── Constraints Adapter  → Conservation areas, green belt, Article 4
    ↓
Data Analysis                → Score, rank, and summarise results
    ↓
Frontend                     → Display results on map + cards
```

## Current Status & Next Steps

**Done:**
- Intent parsing pipeline (Gemini LLM → structured ParsedIntent)
- Location resolver (UK councils, boroughs, neighbourhoods, fuzzy matching)
- Adapter framework with orchestrator (concurrent fan-out, result merging)
- FastAPI with 6 endpoints, works in mock mode with no keys

**Next — plug in IBex API:**
1. Get IBex API key at the hackathon
2. Make a test call, log the response shape
3. Fill in `adapters/ibex.py` — map council names → IBex IDs, wire up real HTTP calls, write `normalize_results`
4. Hit `/api/search` and see real planning data come back

**Then — data analysis layer:**
1. Take merged results from the orchestrator
2. Aggregate: approval rates by borough, trends over time, average decision times
3. Score sites against the user's constraints (flood risk, conservation area, etc.)
4. Return analysis summary + ranked results to the frontend

**Then — frontend integration:**
1. Frontend calls `/api/parse` to show "here's what we understood"
2. User confirms or refines, then frontend calls `/api/search`
3. Display results on a map (Leaflet/Mapbox) with filterable cards
4. Show analysis dashboard: approval rates, price trends, constraint overlays

## Project Structure

```
intent_parser/
├── schema.py              # Domain models (Pydantic v2) — zero API coupling
├── llm_parser.py          # Gemini-powered NLP → ParsedIntent
├── location.py            # UK council/borough/neighbourhood resolver
└── adapters/
    ├── base.py            # Abstract DataSourceAdapter interface
    ├── ibex.py            # IBex planning data (skeleton)
    ├── epc.py             # EPC energy ratings (skeleton)
    ├── flood.py           # EA flood risk zones (skeleton)
    ├── price_paid.py      # HM Land Registry prices (skeleton)
    └── constraints.py     # Conservation areas, green belt, Article 4 (skeleton)

api/
├── main.py                # FastAPI app — 6 endpoints
└── orchestrator.py        # Runs all adapters concurrently, merges results
```

## Quick Start

```bash
pip install -r requirements.txt
cd python
uvicorn api.main:app --reload
```

The server starts in **mock mode** with no API keys needed. Set `GOOGLE_API_KEY` to enable real LLM parsing.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/parse` | Parse a query into structured intent |
| POST | `/api/plan` | Parse + show what API queries *would* be made (dry run) |
| POST | `/api/search` | Full pipeline: parse → search all adapters → return results |
| GET | `/api/councils` | Council list for frontend autocomplete |
| GET | `/api/adapters` | Registered adapters and their status |
| GET | `/api/health` | Health check |

**Example:**

```bash
curl -X POST http://localhost:8000/api/parse \
  -H "Content-Type: application/json" \
  -d '{"query": "Build 20 affordable flats in South London, avoiding flood zones"}'
```

## Data Source Adapters

Each adapter is independent — enable them by setting environment variables:

| Adapter | Env Var | Status |
|---|---|---|
| IBex Planning Data | `IBEX_API_TOKEN` | Skeleton — builds payloads, needs API key |
| EPC Energy Ratings | `EPC_API_TOKEN` | Skeleton — needs signup at epc.opendatacommunities.org |
| Flood Risk Zones | `FLOOD_DATA_DIR` | Skeleton — needs EA GeoJSON download |
| HM Land Registry | `PRICE_PAID_DB` | Skeleton — needs Price Paid CSV → SQLite |
| Planning Constraints | `CONSTRAINTS_DATA_DIR` | Skeleton — needs planning.data.gov.uk GeoJSON |

All adapters return empty results until configured. The system works fine with just mock mode.

## Environment Variables

```bash
# LLM parsing (optional — falls back to keyword-based mock)
GOOGLE_API_KEY=...

# Data sources (all optional)
IBEX_API_TOKEN=...
IBEX_BASE_URL=https://api.ibexenterprise.com
EPC_API_TOKEN=...
FLOOD_DATA_DIR=./data/flood_zones
PRICE_PAID_DB=./data/price_paid.sqlite
CONSTRAINTS_DATA_DIR=./data/constraints
```

## Location Resolver

Built-in UK geography database with no external dependencies:

- All 33 London boroughs + City of London with neighbourhood aliases
- Greater Manchester, West Midlands, South Yorkshire, West Yorkshire, Merseyside councils
- Major cities: Birmingham, Leeds, Liverpool, Bristol, Sheffield, Edinburgh, Glasgow, Cardiff, and more
- Region mapping: "South London" → [Lambeth, Southwark, Lewisham, ...]
- Fuzzy matching with optional `rapidfuzz` support

## Requirements

- Python 3.14+
- See [requirements.txt](requirements.txt)



ScoringResult (dataclass) — the return type, carrying:

stats: BoroughStats — raw borough statistics
base_approval_score: int — approval rate minus constraint penalties, clamped 0–100; this is the starting point for ApprovalPrediction.score
approval_by_type: dict[str, float] — per normalised_application_type approval rate
applied_penalties: list[tuple[str, int]] — e.g. [("Conservation area", 20), ("Article 4", 10)]
BoroughScorer — stateless class with:

Method	What it does
score(applications, borough_name, flags, council_id)	Main entry point; falls back to mock when list is empty
_approval_by_type(apps)	Groups by normalised_application_type, computes approval % per type
_avg_decision_weeks(apps)	application_date → decided_date in weeks; skips missing/implausible
_detect_trend(apps)	Last-2-years vs prior approval rate; returns None if < 3 samples per window
_apply_penalties(rate, flags)	Subtracts flood (−15), conservation area (−20), green belt (−25), article 4 (−10)
infer_constraints(apps)	Classmethod — detects conservation area, listed building, TPO, article 4 from application types and proposal text; flood/green belt left for GIS
_mock_score(borough_name, flags)	Deterministic mock seeded on borough name