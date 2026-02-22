import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from data import query

router = APIRouter(prefix="/api/councils", tags=["councils"])


@router.get("")
def get_councils():
    df = query("council_boundaries")
    result = []
    for _, row in df.iterrows():
        feature = json.loads(row["feature_json"])
        feature["lad_name"] = row["lad_name"]
        # Rename geometry → polygon to match the frontend interface
        if "geometry" in feature:
            feature["polygon"] = feature.pop("geometry")
        result.append(feature)
    return JSONResponse(content=result)
