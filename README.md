# Siteline — Automated Planning

Find the best place to build. Describe your development in plain English and get ranked borough recommendations, approval likelihood predictions, comparable precedents, and downloadable PDF reports — all backed by real UK planning data.

## What It Does

Type a query like:

> "20-unit affordable housing in South London, avoiding flood zones"

Siteline will:

1. **Understand your intent** — parse your description into a structured brief (development type, scale, location preferences, constraints)
2. **Search planning records** — query the IBex planning API across all candidate boroughs concurrently
3. **Score each location** — calculate approval likelihood using historical data, constraint penalties, and a trained neural network
4. **Find precedents** — surface the most similar past applications as comparable evidence
5. **Rank and report** — return boroughs ranked by a composite viability score, with an optional professional PDF

The backend serves at `http://localhost:8000` with interactive API docs at `/docs`.

## How to Use

### Web Interface

Open the frontend in your browser and enjoy!
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

## Additional Data Sources

Additional adapters can be enabled with environment variables. All are optional: the system gracefully degrades when they're absent.

## Disclaimer
This tool provides statistical analysis of publicly available planning application data. It does not constitute legal, financial, or professional planning advice. Planning decisions are made by local planning authorities on a case-by-case basis. Obtain independent professional advice before making investment or development decisions.
