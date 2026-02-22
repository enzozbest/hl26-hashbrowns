import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import query
from data.analysis_data.db import get_analysis, save_analysis
from report.builder import build_report
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

    # Generate reports for the councils the NN ranked.
    council_ids = [s["council_id"] for s in scores]
    background_tasks.add_task(_generate_reports, council_ids, output_dir)

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
