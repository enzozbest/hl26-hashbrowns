import random
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import query
from data.analysis_data.db import get_analysis, save_analysis
from report.builder import build_report
from report.renderer import render_pdf

router = APIRouter(prefix="/api/analyse", tags=["analyse"])

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


class AnalyseRequest(BaseModel):
    council_ids: list[int]
    prompt: str


@router.post("")
def analyse(body: AnalyseRequest, background_tasks: BackgroundTasks):
    if not body.council_ids:
        raise HTTPException(status_code=422, detail="ids must not be empty")

    analysis_id = str(uuid.uuid4())
    output_dir = REPORTS_DIR / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = [
        {
            "council_id": id_,
            "score": round(random.uniform(0, 100), 1)
        }
        for id_ in body.council_ids
    ]

    save_analysis(analysis_id, str(output_dir))

    background_tasks.add_task(_generate_reports, body.council_ids, output_dir)

    return {
        "analysis_id": analysis_id,
        "scores": scores,
    }


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
