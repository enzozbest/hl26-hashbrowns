import asyncio
import logging
import uuid
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import query
from data.analysis_data.db import get_analysis, save_analysis
from report.builder import build_report
from report.models import CouncilPrediction, IndicatorEntry, OraclePrediction
from report.renderer import render_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyse", tags=["analyse"])

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


class AnalyseRequest(BaseModel):
    council_ids: list[int]
    prompt: str


@router.post("")
def analyse(body: AnalyseRequest, background_tasks: BackgroundTasks):
    analysis_id = str(uuid.uuid4())
    output_dir = REPORTS_DIR / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(body.prompt)

    # Call the planning-oracle neural network for approval scores.
    scores = _predict_scores(body.prompt)

    save_analysis(analysis_id, str(output_dir))

    # Fetch real IBex stats for the ranked councils, then generate PDFs
    # with both NN predictions and real statistics baked in.
    council_ids = [s["council_id"] for s in scores]
    background_tasks.add_task(
        _generate_reports_with_stats, council_ids, scores, output_dir
    )

    return {
        "analysis_id": analysis_id,
        "scores": scores,
    }


def _predict_scores(prompt: str) -> list[dict]:
    """Call the planning-oracle pipeline directly and return ranked council scores."""
    import neural_network.inference.api as nn_api

    if nn_api._pipeline is None:
        raise HTTPException(
            status_code=502,
            detail="Planning oracle pipeline not loaded",
        )

    try:
        result = nn_api._pipeline.predict(prompt)
    except Exception as exc:
        logger.error("Planning oracle prediction failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Planning oracle prediction failed: {exc}",
        )

    return [
        {
            "council_id": council.council_id,
            "council_name": council.council_name,
            "score": round(council.score * 100, 1),
            "indicators": [
                {
                    "name": ind.display_name or ind.name,
                    "value": ind.value,
                    "contribution": ind.contribution,
                    "direction": ind.direction,
                }
                for ind in council.indicators[:5]
            ],
        }
        for council in result.top_councils
    ]


@router.get("/{analysis_id}/report")
def get_report(analysis_id: str, id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")

    boundaries = query("council_boundaries")
    row = boundaries[boundaries["council_id"] == id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Council id {id} not found")

    council_name = row.iloc[0]["council_name"]
    slug = council_name.lower().replace(" ", "_")
    pdf_path = Path(analysis["file_path"]) / f"{slug}.pdf"

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report for council {id} not found in analysis '{analysis_id}'"
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{slug}.pdf"'},
    )


def _generate_reports(council_ids: list[int], output_dir: Path) -> None:
    """Fallback — generates PDFs without NN or IBex stats."""
    boundaries = query("council_boundaries")

    for council_id in council_ids:
        row = boundaries[boundaries["council_id"] == council_id]
        if row.empty:
            print(f"[analyse] no boundary found for council_id={council_id}, skipping")
            continue

        council_name = row.iloc[0]["council_name"]
        try:
            report = build_report(council_name)
            pdf_bytes = render_pdf(report)
            slug = council_name.lower().replace(" ", "_")
            (output_dir / f"{slug}.pdf").write_bytes(pdf_bytes)
            print(f"[analyse] generated report: {slug}.pdf")
        except Exception as exc:
            print(f"[analyse] failed for {council_name}: {exc}")


def _generate_reports_with_stats(
    council_ids: list[int],
    nn_scores: list[dict],
    output_dir: Path,
) -> None:
    """Generate PDFs with real IBex stats and NN predictions baked in."""
    boundaries = query("council_boundaries")
    nn_by_id = {s["council_id"]: s for s in nn_scores}

    # Fetch IBex stats for all councils concurrently
    ibex_stats = asyncio.run(_fetch_ibex_stats_batch(council_ids))

    for council_id in council_ids:
        row = boundaries[boundaries["council_id"] == council_id]
        if row.empty:
            print(f"[analyse] no boundary found for council_id={council_id}, skipping")
            continue

        council_name = row.iloc[0]["council_name"]
        nn_data = nn_by_id.get(council_id)
        stats_data = ibex_stats.get(council_id)

        try:
            # Build OraclePrediction from NN output
            oracle_prediction = _build_oracle_prediction(
                nn_data, council_id, council_name
            ) if nn_data else None

            report = build_report(
                council_name,
                oracle_prediction=oracle_prediction,
            )

            # If we have IBex stats, inject them as an extra section
            if stats_data:
                from report.models import Insight, Metric, SectionResult

                approval_raw = stats_data.get("approval_rate", 0)
                approval_pct = round(approval_raw * 100, 1)  # 0–1 → 0–100%
                activity = stats_data.get("activity_level", "unknown")
                homes = stats_data.get("number_of_new_homes_approved", 0)
                decision_times = stats_data.get("average_decision_time", {})

                metrics = [
                    Metric(
                        label="Approval rate",
                        value=round(approval_pct, 1),
                        unit="%",
                        direction="positive" if approval_pct >= 70 else (
                            "negative" if approval_pct < 40 else "neutral"
                        ),
                    ),
                    Metric(
                        label="Development activity",
                        value=activity.title(),
                        direction="positive" if activity == "high" else "neutral",
                    ),
                    Metric(
                        label="New homes approved",
                        value=homes or 0,
                        direction="positive" if (homes or 0) > 100 else "neutral",
                    ),
                ]

                # Add decision times per project type
                for proj_type, days in decision_times.items():
                    if days is not None:
                        weeks = round(days / 7, 1)
                        metrics.append(Metric(
                            label=f"Avg decision ({proj_type})",
                            value=weeks,
                            unit="weeks",
                            direction="positive" if weeks < 13 else (
                                "negative" if weeks > 26 else "neutral"
                            ),
                        ))

                insights = []
                if approval_pct >= 70:
                    insights.append(Insight(
                        text=f"Strong approval rate of {approval_pct:.1f}% indicates a favourable planning environment.",
                        sentiment="positive",
                    ))
                elif approval_pct < 40:
                    insights.append(Insight(
                        text=f"Low approval rate of {approval_pct:.1f}% — proposals face above-average scrutiny.",
                        sentiment="negative",
                    ))
                else:
                    insights.append(Insight(
                        text=f"Moderate approval rate of {approval_pct:.1f}% — outcome depends on proposal quality.",
                        sentiment="neutral",
                    ))

                ibex_section = SectionResult(
                    section_id="ibex_stats",
                    title="Planning Statistics (IBex)",
                    summary=(
                        f"Based on IBex planning data, {council_name} has an approval rate "
                        f"of {approval_pct:.1f}% with {activity} development activity. "
                        f"{homes or 0} new homes have been approved in the reporting period."
                    ),
                    metrics=metrics,
                    insights=insights,
                    data_quality="full",
                    data_source="IBex Enterprise API (real-time)",
                )
                # Insert IBex stats as the first section
                report.sections.insert(0, ibex_section)

            pdf_bytes = render_pdf(report)
            slug = council_name.lower().replace(" ", "_")
            (output_dir / f"{slug}.pdf").write_bytes(pdf_bytes)
            print(f"[analyse] generated report: {slug}.pdf ({len(pdf_bytes) // 1024} KB)")
        except Exception as exc:
            logger.error("[analyse] failed for %s: %s", council_name, exc, exc_info=True)
            print(f"[analyse] failed for {council_name}: {exc}")


def _build_oracle_prediction(
    nn_data: dict,
    council_id: int,
    council_name: str,
) -> OraclePrediction:
    """Convert raw NN score dict into an OraclePrediction for the report builder."""
    raw_score = nn_data.get("score", 50) / 100.0  # NN returns 0–100, model wants 0–1

    indicators = [
        IndicatorEntry(
            name=ind["name"],
            value=ind["value"],
            contribution=ind["contribution"],
            direction=ind["direction"],
        )
        for ind in nn_data.get("indicators", [])
    ]

    return OraclePrediction(
        approval_probability=raw_score,
        confidence_interval=(max(0, raw_score - 0.12), min(1, raw_score + 0.12)),
        top_councils=[
            CouncilPrediction(
                council_id=council_id,
                council_name=council_name,
                score=raw_score,
                indicators=indicators,
            )
        ],
    )


async def _fetch_ibex_stats_batch(
    council_ids: list[int],
) -> dict[int, dict]:
    """Fetch IBex stats for multiple councils concurrently.

    Returns a dict mapping council_id → stats dict. Councils that fail
    are silently omitted.
    """
    from hashbrowns.config import settings as ibex_settings
    from hashbrowns.ibex.client import IbexClient

    today = date.today()
    dt_from = (today - timedelta(days=365)).isoformat()
    dt_to = today.isoformat()

    results: dict[int, dict] = {}

    try:
        async with IbexClient(ibex_settings) as client:
            async def _fetch_one(cid: int) -> tuple[int, dict | None]:
                try:
                    stats = await client.stats(cid, dt_from, dt_to)
                    return cid, {
                        "approval_rate": stats.approval_rate,
                        "refusal_rate": stats.refusal_rate,
                        "activity_level": stats.council_development_activity_level.value,
                        "average_decision_time": stats.average_decision_time.model_dump(
                            by_alias=True, exclude_none=True,
                        ),
                        "number_of_applications": stats.number_of_applications.model_dump(
                            by_alias=True, exclude_none=True,
                        ),
                        "number_of_new_homes_approved": stats.number_of_new_homes_approved,
                    }
                except Exception as exc:
                    logger.warning("IBex stats failed for council %d: %s", cid, exc)
                    return cid, None

            fetched = await asyncio.gather(*[_fetch_one(cid) for cid in council_ids])
            for cid, data in fetched:
                if data is not None:
                    results[cid] = data
    except Exception as exc:
        logger.error("IBex client setup failed: %s", exc)

    return results
