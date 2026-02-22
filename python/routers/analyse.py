import logging
import os
import uuid
from pathlib import Path

import httpx
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
ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:8001")


class AnalyseRequest(BaseModel):
    prompt: str


@router.post("")
def analyse(body: AnalyseRequest, background_tasks: BackgroundTasks):
    analysis_id = str(uuid.uuid4())
    output_dir = REPORTS_DIR / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

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
    """Call the planning-oracle /predict endpoint and return ranked council scores."""
    try:
        resp = httpx.post(
            f"{ORACLE_URL}/predict",
            json={"proposal_text": prompt},
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Planning oracle request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Planning oracle unavailable: {exc}",
        )

    data = resp.json()
    result = data.get("result", {})

    # Return the NN's ranked councils with scores scaled to 0-100.
    return [
        {
            "council_id": council["council_id"],
            "score": round(council["score"] * 100, 1),
        }
        for council in result.get("top_councils", [])
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
