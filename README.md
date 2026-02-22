# Hashbrowns — Planning Site Finder

Find the best place to build. Describe your development in plain English and get ranked borough recommendations, approval likelihood predictions, comparable precedents, and downloadable PDF reports — all backed by real UK planning data.

## What It Does

Type a query like:

> "20-unit affordable housing in South London, avoiding flood zones"

Hashbrowns will:

1. **Understand your intent** — parse your description into a structured brief (development type, scale, location preferences, constraints)
2. **Search planning records** — query the IBex planning API across all candidate boroughs concurrently
3. **Score each location** — calculate approval likelihood using historical data, constraint penalties, and a trained neural network
4. **Find precedents** — surface the most similar past applications as comparable evidence
5. **Rank and report** — return boroughs ranked by a composite viability score, with an optional professional PDF

## Quick Start

### Prerequisites

- Python 3.14+
- Node.js (for the frontend)
- [uv](https://docs.astral.sh/uv/) or pip

### 1. Install

```bash
# Backend
cd python
pip install -e ".[dev]"

# Frontend
cd ../ts
npm install
```

### 2. Configure

Create a `.env` file in `python/`:

```dotenv
# Required — IBex Enterprise API key
IBEX_API_KEY=your_key_here

# Optional — enables real NLP parsing (falls back to rule-based mock without it)
GOOGLE_API_KEY=your_google_key_here
```

No API keys? No problem — the system runs in **mock mode** with synthetic but deterministic data, so you can explore the full workflow without any credentials.

### 3. Run

```bash
# Start the backend (from python/)
uvicorn api.main:app --reload --port 8000

# Start the frontend (from ts/)
npm run dev
```

The backend serves at `http://localhost:8000` with interactive API docs at `/docs`.

## How to Use

### Web Interface

Open the frontend in your browser. Type a natural-language development query on the home page, and view ranked results on an interactive map.

### API

| Endpoint | What it does |
|---|---|
| `POST /api/parse` | Parse a query into structured intent |
| `POST /api/report` | Full pipeline — returns a ranked JSON report |
| `POST /api/report/pdf` | Same analysis, returned as a downloadable PDF |
| `POST /api/search` | Raw search results from all data adapters |
| `GET /api/health` | Health check |

**Example — generate a report:**

```bash
curl -X POST http://localhost:8000/api/report \
  -H "Content-Type: application/json" \
  -d '{"query": "20-unit affordable housing in South London"}'
```

**Example — download a PDF:**

```bash
curl -X POST http://localhost:8000/api/report/pdf \
  -H "Content-Type: application/json" \
  -d '{"query": "20-unit affordable housing in South London"}' \
  -o report.pdf
```

## What's in the Report

Each borough report includes:

- **Viability score** (0--100) with a high/medium/low band
- **Approval prediction** with confidence interval and risk/positive factors
- **Borough statistics** — total applications, approval rate, average decision time, trend direction
- **Constraint flags** — conservation area, flood risk, green belt, Article 4
- **Top comparable applications** — the most similar historical precedents with similarity scores and explanations
- **Summary and key considerations** in plain language

PDF reports add a cover page, executive summary, per-borough detail pages, and a methodology section.

## How Scoring Works

The viability score blends four components:

| Component | Weight | What it measures |
|---|---|---|
| Approval prediction | 50% | Historical approval rate adjusted for constraints |
| Comparable evidence | 20% | How similar past applications fared |
| Decision speed | 15% | How quickly the council decides (faster = better) |
| Borough activity | 15% | Volume of planning applications (more data = more reliable) |

Constraint penalties are applied for flood zones (-15), conservation areas (-20), green belt (-25), and Article 4 directions (-10).

## ML Prediction Model

The planning oracle is a multi-branch neural network that predicts approval probability for a given proposal:

- **Text branch** — encodes the proposal description using sentence-transformer embeddings
- **Application branch** — processes structured metadata (unit count, floor area, development type)
- **Council branch** — incorporates borough-level statistics (approval rate, decision speed)

These are fused and output a calibrated approval probability with confidence interval.

Train the model with:

```bash
cd python
make train COUNCILS="366 66 77 273 320"
```

## Project Structure

```
hl26-hashbrowns/
├── ts/                     Frontend (React, Leaflet, TailwindCSS, Vite)
├── python/
│   ├── api/                FastAPI endpoints
│   ├── intent_parser/      NLP query parsing (Gemini LLM + rule-based fallback)
│   ├── analysis/           Due diligence engine, scoring, comparable finder, PDF generation
│   ├── hashbrowns/         IBex API client and shared config
│   ├── neural_network/     ML model (training, inference, features)
│   ├── data/               Council databases, region mappings
│   └── Makefile            Dev shortcuts (train, serve, test)
└── planning-oracle/        Standalone ML module (merged into python/)
```

## Optional Data Sources

Additional adapters can be enabled with environment variables. All are optional — the system gracefully degrades when they're absent.

| Data Source | Env Var | What it provides |
|---|---|---|
| EPC Ratings | `EPC_API_TOKEN` | Energy performance certificates |
| Flood Risk | `FLOOD_DATA_DIR` | EA flood zone GeoJSON overlays |
| Land Registry | `PRICE_PAID_DB` | Historical property transaction prices |
| Constraints | `CONSTRAINTS_DATA_DIR` | Conservation areas, green belt, Article 4 GeoJSON |

## Disclaimer

This tool provides statistical analysis of publicly available planning application data. It does not constitute legal, financial, or professional planning advice. Planning decisions are made by local planning authorities on a case-by-case basis. Obtain independent professional advice before making investment or development decisions.
