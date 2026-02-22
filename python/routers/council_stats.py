"""Router for /api/council-stats — fetches real IBex planning statistics."""

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hashbrowns.config import settings as ibex_settings
from hashbrowns.ibex.client import IbexClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["council-stats"])

# Initialised at startup, torn down at shutdown.
_ibex_client: IbexClient | None = None


async def startup() -> None:
    """Open the IBex HTTP client.  Called from main.py @startup."""
    global _ibex_client
    _ibex_client = IbexClient(ibex_settings)
    await _ibex_client.__aenter__()
    logger.info("IBex client ready (council-stats router)")


async def shutdown() -> None:
    """Close the IBex HTTP client.  Called from main.py @shutdown."""
    global _ibex_client
    if _ibex_client is not None:
        await _ibex_client.__aexit__(None, None, None)
        _ibex_client = None


class CouncilStatsRequest(BaseModel):
    council_ids: list[int]
    date_from: str | None = None
    date_to: str | None = None


@router.post("/council-stats")
async def council_stats(body: CouncilStatsRequest) -> dict[str, Any]:
    """Fetch real IBex statistics for a list of council IDs."""
    if _ibex_client is None:
        raise HTTPException(status_code=503, detail="IBex client not ready")

    today = date.today()
    dt_to = body.date_to or today.isoformat()
    dt_from = body.date_from or (today - timedelta(days=365)).isoformat()

    async def _fetch_one(council_id: int) -> tuple[int, dict[str, Any] | str]:
        try:
            stats = await _ibex_client.stats(council_id, dt_from, dt_to)
            return council_id, {
                "council_id": council_id,
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
            logger.warning("Stats fetch failed for council %d: %s", council_id, exc)
            return council_id, f"Council {council_id}: {exc}"

    results = await asyncio.gather(*[_fetch_one(cid) for cid in body.council_ids])

    council_stats_map: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for council_id, outcome in results:
        if isinstance(outcome, str):
            errors.append(outcome)
        else:
            council_stats_map[str(council_id)] = outcome

    return {"council_stats": council_stats_map, "errors": errors}
