import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyse, council_stats, councils

logger = logging.getLogger(__name__)

app = FastAPI(title="Hashbrowns Planning Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(councils.router)
app.include_router(analyse.router)
app.include_router(council_stats.router)

import neural_network.inference.api as _nn_api  # noqa: E402

app.post("/predict", response_model=_nn_api.PredictionResponse)(_nn_api.predict)


@app.on_event("startup")
async def _load_oracle_pipeline() -> None:
    """Load the planning-oracle inference pipeline at server start."""
    try:
        _nn_api._pipeline = _nn_api._load_pipeline()
        logger.info("Planning oracle pipeline loaded")
    except Exception as exc:
        logger.error("Failed to load oracle pipeline: %s", exc)
        _nn_api._pipeline = None


@app.on_event("startup")
async def _start_ibex_client() -> None:
    await council_stats.startup()


@app.on_event("shutdown")
async def _stop_ibex_client() -> None:
    await council_stats.shutdown()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
