"""End-to-end pipeline: train → calibrate → evaluate → serve.

Usage examples:
    # Full pipeline
    python -m scripts.run_pipeline --councils council-01 council-02

    # Train and evaluate only
    python -m scripts.run_pipeline --councils council-01 --skip-calibrate

    # Run everything then start the server
    python -m scripts.run_pipeline --councils council-01 --serve
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _elapsed(start: float) -> str:
    secs = time.time() - start
    mins, secs = divmod(int(secs), 60)
    return f"{mins}m {secs}s"


# ── Stage 1: Train ──────────────────────────────────────────────────────────


def run_train(
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_dir: str,
    council_ids: list[int],
    date_from: str,
    date_to: str,
) -> dict:
    from config.settings import Settings
    from data.api_client import PlanningAPIClient
    from training.train import train_model

    t0 = time.time()
    log.info("Starting training (%d epochs, batch_size=%d, lr=%.1e) …", epochs, batch_size, lr)

    config = Settings(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        checkpoint_dir=checkpoint_dir,
    )

    async def _run() -> tuple:
        async with PlanningAPIClient(settings=config) as client:
            return await train_model(
                config, client,
                council_ids=council_ids,
                date_from=date_from,
                date_to=date_to,
            )

    _, metrics = asyncio.run(_run())

    log.info("Training finished in %s", _elapsed(t0))
    return metrics


# ── Stage 2: Calibrate ──────────────────────────────────────────────────────


def run_calibrate(
    checkpoint_dir: str,
    council_ids: list[int],
    date_from: str,
    date_to: str,
) -> None:
    import torch

    from model.approval_model import load_model
    from model.calibration import TemperatureScaler
    from training.dataset import build_dataloaders, build_datasets

    from config.settings import Settings
    from data.api_client import PlanningAPIClient

    t0 = time.time()
    log.info("Calibrating model (temperature scaling) …")

    config = Settings(checkpoint_dir=checkpoint_dir)

    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        log.warning("No checkpoint at %s — skipping calibration", checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, weights_only=True)

    from features.application import ApplicationFeatureExtractor
    from features.council import CouncilFeatureExtractor
    from features.text import TextEmbedder

    text_embedder = TextEmbedder()
    app_extractor = ApplicationFeatureExtractor.load(
        Path(checkpoint_dir) / "app_extractor.pkl",
    )
    council_extractor = CouncilFeatureExtractor.load(
        Path(checkpoint_dir) / "council_extractor.pkl",
    )

    async def _build():
        async with PlanningAPIClient(settings=config) as client:
            return await build_datasets(
                client, text_embedder, app_extractor, council_extractor,
                council_ids=council_ids,
                date_from=date_from,
                date_to=date_to,
            )

    _, val_ds, _ = asyncio.run(_build())
    _, val_loader, _ = build_dataloaders(val_ds, val_ds, val_ds, batch_size=config.batch_size)

    model = load_model(
        checkpoint_path,
        text_dim=checkpoint.get("text_dim", config.embedding_dim),
        app_dim=checkpoint.get("app_dim", 15),
        council_dim=checkpoint.get("council_dim", 10),
    )

    scaler = TemperatureScaler()
    final_t = scaler.fit(model, val_loader)
    scaler.save(Path(checkpoint_dir) / "temperature_scaler.pkl")
    log.info("Calibration done (T=%.4f) in %s", final_t, _elapsed(t0))


# ── Stage 3: Evaluate ──────────────────────────────────────────────────────


def run_evaluate(
    checkpoint_dir: str,
    output_dir: str,
    council_ids: list[int],
    date_from: str,
    date_to: str,
) -> dict:
    import torch

    from model.approval_model import load_model
    from model.calibration import TemperatureScaler
    from training.dataset import build_dataloaders, build_datasets
    from training.evaluate import evaluate_model

    from config.settings import Settings
    from data.api_client import PlanningAPIClient

    t0 = time.time()
    log.info("Evaluating model …")

    config = Settings(checkpoint_dir=checkpoint_dir, output_dir=output_dir)

    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    from features.application import ApplicationFeatureExtractor
    from features.council import CouncilFeatureExtractor
    from features.text import TextEmbedder

    text_embedder = TextEmbedder()
    app_extractor = ApplicationFeatureExtractor.load(
        Path(checkpoint_dir) / "app_extractor.pkl",
    )
    council_extractor = CouncilFeatureExtractor.load(
        Path(checkpoint_dir) / "council_extractor.pkl",
    )

    async def _build():
        async with PlanningAPIClient(settings=config) as client:
            return await build_datasets(
                client, text_embedder, app_extractor, council_extractor,
                council_ids=council_ids,
                date_from=date_from,
                date_to=date_to,
            )

    _, _, test_ds = asyncio.run(_build())
    _, _, test_loader = build_dataloaders(test_ds, test_ds, test_ds, batch_size=config.batch_size)

    model = load_model(
        checkpoint_path,
        text_dim=checkpoint.get("text_dim", config.embedding_dim),
        app_dim=checkpoint.get("app_dim", 15),
        council_dim=checkpoint.get("council_dim", 10),
    )

    scaler_path = Path(checkpoint_dir) / "temperature_scaler.pkl"
    scaler = TemperatureScaler.load(scaler_path) if scaler_path.exists() else None

    metrics = evaluate_model(model, scaler, test_loader, output_dir=output_dir)

    log.info("── Evaluation Summary ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info("  %-20s  %.4f", k, v)

    log.info("Evaluation finished in %s", _elapsed(t0))
    return metrics


# ── Stage 4: Serve ──────────────────────────────────────────────────────────


def run_serve(port: int) -> None:
    import uvicorn

    log.info("Starting API server on port %d …", port)
    uvicorn.run("inference.api:app", host="0.0.0.0", port=port, reload=False)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Planning Oracle — end-to-end ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Stage flags
    p.add_argument("--skip-train", action="store_true", help="Skip training")
    p.add_argument("--skip-calibrate", action="store_true", help="Skip calibration")
    p.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation")
    p.add_argument("--serve", action="store_true", help="Start API server after pipeline")

    # Data source options
    p.add_argument(
        "--councils", nargs="+", type=int, default=[],
        help="Council IDs (integers) to fetch data for",
    )
    p.add_argument("--date-from", default="2020-01-01", help="Search start date (default: 2020-01-01)")
    p.add_argument("--date-to", default="2025-12-31", help="Search end date (default: 2025-12-31)")

    # Training options
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="outputs")

    # Serve options
    p.add_argument("--port", type=int, default=8000)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(exist_ok=True)

    # Full pipeline
    stages = []
    if not args.skip_train:
        stages.append("train")
    if not args.skip_calibrate:
        stages.append("calibrate")
    if not args.skip_evaluate:
        stages.append("evaluate")
    if args.serve:
        stages.append("serve")

    log.info("Pipeline stages: %s", " → ".join(stages))

    if "train" in stages:
        run_train(
            args.epochs, args.batch_size, args.lr, args.checkpoint_dir,
            args.councils, args.date_from, args.date_to,
        )

    if "calibrate" in stages:
        run_calibrate(
            args.checkpoint_dir, args.councils, args.date_from, args.date_to,
        )

    if "evaluate" in stages:
        run_evaluate(
            args.checkpoint_dir, args.output_dir,
            args.councils, args.date_from, args.date_to,
        )

    log.info("Pipeline complete in %s", _elapsed(t0))

    if "serve" in stages:
        run_serve(args.port)


if __name__ == "__main__":
    main()
