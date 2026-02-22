from fastapi import APIRouter
from fastapi.responses import Response

from data import query

router = APIRouter(prefix="/api/councils", tags=["councils"])


@router.get("")
def get_councils():
    df = query("council_boundaries")
    return Response(
        content="[" + ",".join(df["feature_json"]) + "]",
        media_type="application/json",
    )
