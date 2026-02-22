"""FastAPI endpoint exposing the prediction pipeline.

Provides a POST ``/predict`` endpoint that accepts a natural-language
planning proposal and returns a structured prediction with approval
probability, council rankings, and feature attributions.

Start the server::

    uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from inference.pipeline import InferencePipeline, PredictionResult

logger = logging.getLogger(__name__)

# ── Lazy import to avoid hard FastAPI dependency at module level ──────────
from fastapi import FastAPI, HTTPException


# ── Request / response models ────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Request body for the ``/predict`` endpoint."""

    proposal_text: str = Field(
        ...,
        min_length=10,
        description="Free-text description of the planning proposal",
    )


class PredictionResponse(BaseModel):
    """Response body for the ``/predict`` endpoint."""

    result: PredictionResult = Field(
        ..., description="Full prediction result",
    )


# ── Pipeline state ───────────────────────────────────────────────────────────

_pipeline: Optional[InferencePipeline] = None


def _load_pipeline(settings: Optional[Settings] = None) -> InferencePipeline:
    """Construct the inference pipeline from saved artefacts.

    Loads the model checkpoint, calibrator, feature extractors, and
    council stats from the configured checkpoint directory.

    Args:
        settings: Application settings (defaults to ``get_settings()``).

    Returns:
        A ready-to-use :class:`InferencePipeline`.
    """
    import json

    import torch

    from data.schema import CouncilStats
    from features.application import ApplicationFeatureExtractor
    from features.council import CouncilFeatureExtractor
    from features.text import TextEmbedder
    from inference.parser import ProposalParser
    from model.approval_model import ApprovalModel, load_model
    from model.calibration import TemperatureScaler
    from model.council_ranker import CouncilRanker

    settings = settings or get_settings()
    ckpt_dir = Path(settings.checkpoint_dir)

    # ── Load model ───────────────────────────────────────────────
    ckpt_path = ckpt_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = ApprovalModel(
        text_embed_dim=ckpt["text_embed_dim"],
        num_app_features=ckpt["num_app_features"],
        num_council_features=ckpt["num_council_features"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)

    # ── Load calibrator ──────────────────────────────────────────
    scaler_path = ckpt_dir / "temperature_scaler.pkl"
    if scaler_path.exists():
        calibrator = TemperatureScaler.load(scaler_path)
    else:
        logger.warning("No temperature scaler found — using default T=1.0")
        calibrator = TemperatureScaler()

    # ── Load feature extractors ──────────────────────────────────
    app_ext_path = ckpt_dir / "app_extractor.pkl"
    council_ext_path = ckpt_dir / "council_extractor.pkl"

    app_extractor = ApplicationFeatureExtractor.load(app_ext_path)
    council_extractor = CouncilFeatureExtractor.load(council_ext_path)
    logger.info("Feature extractors loaded")

    # ── Load council stats from JSON artefact ────────────────────
    council_stats: dict[int, CouncilStats] = {}
    stats_path = ckpt_dir / "council_stats.json"
    try:
        raw = json.loads(stats_path.read_text())
        for entry in raw:
            cs = CouncilStats(**entry)
            if cs.council_id is not None:
                council_stats[cs.council_id] = cs
        logger.info("Loaded %d council stats from %s", len(council_stats), stats_path)
    except FileNotFoundError:
        logger.warning(
            "No council_stats.json found in %s — council ranking will be empty",
            ckpt_dir,
        )

    # Back-fill council_name from the SQLite cache when the checkpoint
    # was saved without names (the stats endpoint doesn't return them).
    missing_names = [cs for cs in council_stats.values() if not cs.council_name]
    if missing_names:
        try:
            import sqlite3

            db_path = Path(__file__).resolve().parents[2] / "python" / "data" / "ibex_data" / "ibex.db"
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                rows = conn.execute(
                    "SELECT DISTINCT council_id, council_name "
                    "FROM ibex_applications WHERE council_name IS NOT NULL",
                ).fetchall()
                conn.close()
                name_lookup = {cid: name for cid, name in rows}
                n_names = 0
                for cs in missing_names:
                    name = name_lookup.get(cs.council_id)
                    if name:
                        cs.council_name = name
                        n_names += 1
                if n_names:
                    logger.info("Back-filled council_name for %d councils from DB", n_names)
        except Exception:
            logger.debug("Could not back-fill council names from DB")

    # Back-fill region for council stats that were saved before the
    # region field was added.
    try:
        from data.regions import resolve_council_region

        n_filled = 0
        for cs in council_stats.values():
            if cs.region is None and cs.council_name:
                cs.region = resolve_council_region(cs.council_name)
                if cs.region:
                    n_filled += 1
        if n_filled:
            logger.info("Back-filled region for %d councils", n_filled)
    except ImportError:
        logger.debug("data.regions not available — skipping region back-fill")

    # ── Assemble pipeline ────────────────────────────────────────
    return InferencePipeline(
        parser=ProposalParser(),
        text_embedder=TextEmbedder(settings=settings),
        app_extractor=app_extractor,
        council_extractor=council_extractor,
        council_ranker=CouncilRanker(),
        model=model,
        calibrator=calibrator,
        council_stats=council_stats,
        settings=settings,
    )


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the inference pipeline on startup."""
    global _pipeline
    try:
        _pipeline = _load_pipeline()
        logger.info("Inference pipeline ready")
    except FileNotFoundError as exc:
        logger.error(
            "Cannot start: model artefacts not found (%s). "
            "Run training first.",
            exc,
        )
        _pipeline = None
    yield
    _pipeline = None


# ── App ──────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Planning Oracle",
    description="Predict planning application approval probability",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint."""
    status = "ok" if _pipeline is not None else "degraded"
    return {"status": status}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate an approval probability prediction for a proposal.

    Args:
        request: Contains the free-text proposal description.

    Returns:
        Structured prediction with probability, council rankings,
        and feature attributions.

    Raises:
        HTTPException: If the pipeline is not initialised or prediction fails.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialised. Run training first.",
        )
    try:
        result = _pipeline.predict(request.proposal_text)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(result=result)
