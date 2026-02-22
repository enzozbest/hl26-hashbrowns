from fastapi import APIRouter
from shapely import wkb
from shapely.geometry import mapping

from data import query

router = APIRouter(prefix="/api/councils", tags=["councils"])


@router.get("")
def get_councils():
    df = query("council_boundaries")

    features = []
    for _, row in df.iterrows():
        geom = wkb.loads(bytes(row["geometry"])) if row["geometry"] is not None else None
        features.append({
            "ons_code": row["ons_code"],
            "council_name": row["council_name"],
            "council_id": int(row["council_id"]),
            "geometry": mapping(geom) if geom is not None else None,
        })

    return features
