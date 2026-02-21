# Due Diligence Agent & PDF Reporting — Usage Guide

## What This Feature Does

The due diligence pipeline takes a plain-English development brief (e.g. *"20-unit affordable housing in South London"*), parses it into a structured intent, queries the IBex planning application API across all candidate boroughs concurrently, scores each borough on approval likelihood, selects comparable precedent applications, ranks the boroughs by a composite viability score, and returns structured `SiteViabilityReport` objects — optionally rendered as a professional PDF report suitable for client delivery.

---

## Prerequisites & Installation

| Requirement | Version |
|---|---|
| Python | ≥ 3.14 |
| Package manager | `pip` or any PEP 621-compatible tool (`uv`, `pdm`, etc.) |

### 1. Clone and install

```bash
cd hl26-hashbrowns/python
pip install -e ".[dev]"          # installs runtime + test dependencies from pyproject.toml
# or, if using the flat requirements file:
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in `python/` (or export variables directly):

```dotenv
# Required — IBex Enterprise API
IBEX_API_KEY=your_ibex_api_key_here

# Optional — enables real LLM intent parsing (otherwise mock parser is used)
GOOGLE_API_KEY=your_google_api_key_here

# Optional — data-source adapters (gracefully degrade when absent)
IBEX_API_TOKEN=your_ibex_api_token_here
IBEX_BASE_URL=https://api.ibexenterprise.com
EPC_API_TOKEN=your_epc_api_token_here
FLOOD_DATA_DIR=/path/to/flood/data
PRICE_PAID_DB=/path/to/price_paid.db
CONSTRAINTS_DATA_DIR=/path/to/constraints/data
```

> **Note:** When `GOOGLE_API_KEY` is not set, the parser falls back to a deterministic mock that handles common query patterns. When `IBEX_API_KEY` points to an unreachable server, the scorer falls back to mock mode with synthetic (but deterministic) statistics.

---

## Running the API Locally

```bash
cd python
uvicorn api.main:app --reload --port 8000
```

The server starts at `http://localhost:8000`. Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/parse` | Parse a query into a `ParsedIntent` |
| `POST` | `/api/plan` | Dry-run: parse + show planned API queries |
| `POST` | `/api/search` | Full search pipeline across all adapters |
| `POST` | `/api/report` | Due diligence report (JSON) |
| `POST` | `/api/report/pdf` | Due diligence report (PDF download) |
| `GET` | `/api/councils` | Council lookup for autocomplete |
| `GET` | `/api/health` | Health check |

---

## Using `/api/report` (JSON)

### Request

```bash
curl -X POST http://localhost:8000/api/report \
  -H "Content-Type: application/json" \
  -d '{"query": "20-unit affordable housing in South London"}'
```

### Response shape

```jsonc
{
  "reports": [
    {
      "borough": "Lambeth",
      "council_id": 240,
      "rank": 1,
      "recommended": true,
      "viability_score": 72,          // 0–100 composite
      "viability_band": "high",       // "high" | "medium" | "low"
      "approval_prediction": {
        "score": 68,                   // 0–100
        "confidence": 0.8,             // 0.0–1.0
        "comparable_approval_rate": 75.0,
        "num_comparables": 5,
        "borough_baseline_rate": 71.2,
        "risk_factors": [
          {
            "label": "Conservation area",
            "description": "Conservation area constraint applies — may require additional consent.",
            "score_impact": -20,
            "category": "constraint"
          }
        ],
        "positive_factors": [ /* ... */ ],
        "verdict": "Moderate approval likelihood — conservation area is the main risk."
      },
      "borough_stats": {
        "name": "Lambeth",
        "total_applications": 142,
        "approved": 101,
        "refused": 30,
        "approval_rate": 77.1,
        "avg_decision_weeks": 11.3,
        "trend": "improving",
        "trend_detail": "80% approval in last 2 years vs 72% prior",
        "data_quality": "full"         // "full" | "partial" | "mock"
      },
      "constraint_flags": {
        "conservation_area": true,
        "flood_risk": false,
        "green_belt": false,
        "article_4": false
        // ... other boolean flags
      },
      "comparable_applications": [
        {
          "planning_reference": "21/01234/FUL",
          "council_name": "Lambeth",
          "normalised_decision": "Approved",
          "similarity_score": 0.82,
          "similarity_reasons": [
            "Same application type (full planning application)",
            "Same borough (Lambeth)",
            "Similar scale (18 units vs 20 requested)"
          ]
        }
      ],
      "summary": "Lambeth has a 77% approval rate across 142 applications. Approval likelihood is moderate (score: 68/100). Key constraints: Conservation area. Average decision time: 11 weeks.",
      "key_considerations": [
        "Conservation area applies (-20 pts)",
        "Comparable approval rate: 75% (5 precedents)",
        "Fast LPA: avg 11 weeks to decision"
      ],
      "data_quality": "full"
    }
    // ... more boroughs, ranked by viability_score descending
  ],
  "count": 3,
  "top_borough": "Lambeth"
}
```

---

## Using `/api/report/pdf` (PDF Download)

```bash
curl -X POST http://localhost:8000/api/report/pdf \
  -H "Content-Type: application/json" \
  -d '{"query": "20-unit affordable housing in South London"}' \
  -o due_diligence_report.pdf
```

The response has `Content-Type: application/pdf` and a `Content-Disposition: attachment` header. The PDF includes:

1. **Cover page** — title, query, date, borough count
2. **Executive summary** — top recommendation, summary table of all boroughs
3. **Per-borough detail pages** — viability score card, approval prediction verdict, risk/opportunity factors table, constraint flags, top 3 comparable applications, estimated decision timeline
4. **Methodology** — explanation of scoring dimensions and data quality
5. **Disclaimer** — *"statistical analysis, not legal advice"*

---

## Explanation of Scoring

### Approval Rate

For each borough, the raw approval rate is computed from all decided applications (Approved + Refused) fetched from IBex:

```
approval_rate = approved / (approved + refused) × 100
```

Per-type rates are also computed (e.g. full planning applications vs householder applications).

### Average Decision Time

Calculated as the mean of `(decided_date − application_date)` across all applications with both dates present, expressed in **weeks**. Records with negative deltas or gaps exceeding 3 years are excluded as data anomalies.

### Approval Trend

Applications are split into two windows using `decided_date`:

- **Recent:** last 2 years (730 days)
- **Prior:** everything before that

If both windows have ≥ 3 decided applications, the approval rates are compared:

| Difference | Direction |
|---|---|
| > +5 pp | `improving` |
| < −5 pp | `declining` |
| otherwise | `stable` |

### Constraint Penalties

The base approval score starts at the raw approval rate and is reduced by:

| Constraint | Penalty |
|---|---|
| Flood zone | −15 pts |
| Conservation area | −20 pts |
| Green Belt | −25 pts |
| Article 4 Direction | −10 pts |

Penalties are cumulative. The result is clamped to **0–100**.

Constraint flags are inferred heuristically from application metadata and proposal text. Flood zone and Green Belt flags require external GIS data for accuracy.

### Composite Viability Score

The viability score is a weighted average of four components, each scored 0–100:

| Component | Weight | How it's calculated |
|---|---|---|
| Approval prediction | 50% | Base approval score blended with comparable outcome rate |
| Comparable evidence | 20% | Mean similarity score of selected comparables (× 100) |
| Decision speed | 15% | `100 − (avg_weeks − 8) × 5`, clamped 0–100 |
| Borough activity | 15% | `min(total_applications, 100)` |

The weighted sum is rounded and clamped to **0–100**.

### Viability Band

| Score | Band |
|---|---|
| ≥ 70 | `high` |
| 40–69 | `medium` |
| < 40 | `low` |

---

## Comparable Selection Logic

The `ComparableFinder` selects the most relevant **decided** (Approved or Refused) applications from the IBex results as precedent evidence. Each candidate is scored on six dimensions:

1. **Application type match (30%)** — Does the IBex `normalised_application_type` match what the user's development category expects? (e.g. residential → full planning application). Exact match = 1.0, related type = 0.2.

2. **Borough match (25%)** — Is the application in one of the user's target boroughs? Same borough = 1.0, different = 0.1.

3. **Unit count proximity (20%)** — How close is the application's unit count to the requested number? Score = `1 − |diff| / target`. Similar scale (≥ 70%) earns a reason label.

4. **Project type match (10%)** — Does the IBex `project_type` (e.g. `small_residential`, `medium_residential`) match the expected category?

5. **Recency (10%)** — Linear decay over a 5-year window. An application decided today scores 1.0; one decided 5 years ago scores 0.0.

6. **Keyword overlap (5%)** — Do any of the user's keywords or tags appear in the application's proposal text?

The top 5 candidates (by weighted sum) are returned as `ComparableApplication` objects, each with a `similarity_score` (0–1) and a list of human-readable `similarity_reasons`.

---

## Programmatic Usage (Python)

You can use the agent directly without the API:

```python
import asyncio
from hashbrowns.config import Settings
from analysis.agent import DueDiligenceAgent
from analysis.report_generator import generate_report
from pathlib import Path

async def main():
    settings = Settings()  # reads .env
    agent = DueDiligenceAgent(settings)

    async with agent:
        reports = await agent.run_from_intent(parsed_intent)

    for r in reports:
        print(f"#{r.rank} {r.borough}: {r.viability_score}/100 ({r.viability_band})")

    # Generate PDF
    pdf_bytes = generate_report(reports, query="20-unit housing in South London")
    Path("report.pdf").write_bytes(pdf_bytes)
    print(f"PDF written: {len(pdf_bytes):,} bytes")

asyncio.run(main())
```

---

## Notes, Limitations & Disclaimer

- **Mock mode:** When `IBEX_API_KEY` is not configured or IBex returns no data for a borough, the scorer generates deterministic synthetic statistics seeded on the borough name. Reports will show `data_quality: "mock"` — these are placeholders, not real planning data.

- **Constraint inference is heuristic:** Conservation area and listed building flags are inferred from application types and proposal text keywords. Flood zone and Green Belt flags **cannot** be reliably detected from application data alone and require external GIS constraint layers.

- **Coordinate conversion:** The agent uses a linear approximation for WGS84 → OSGB36 conversion (centred on London). For sites far from London, consider using `pyproj` for sub-metre accuracy.

- **LLM parser fallback:** Without `GOOGLE_API_KEY`, the intent parser uses a rule-based mock. Complex or ambiguous queries may not parse correctly in mock mode.

- **Rate limits:** The IBex API may impose rate limits. The agent's concurrency is bounded by `ibex_max_concurrency` in Settings (default: 10).

### Disclaimer

> **This tool provides statistical analysis of publicly available planning application data. It does not constitute legal, financial, or professional planning advice.** Planning decisions are made by local planning authorities on a case-by-case basis, taking into account material considerations that may not be captured in historical data. Recipients should obtain independent professional advice from a qualified planning consultant before making any investment, acquisition, or development decisions. The authors accept no liability for loss or damage arising from reliance on this analysis.
