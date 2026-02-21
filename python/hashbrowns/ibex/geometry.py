"""SRID-aware geometry parsing utilities for Ibex API responses."""
from pyproj import Transformer

# Module-level transformers — instantiated once for performance
_BNG_TO_WGS84 = Transformer.from_crs(27700, 4326, always_xy=True)

_UK_LAT_MIN, _UK_LAT_MAX = 49.8, 60.9
_UK_LON_MIN, _UK_LON_MAX = -8.6, 1.8


def parse_point_wkt(wkt: str, source_srid: int) -> tuple[float, float]:
    """Parse a POINT WKT string and return (longitude, latitude) in WGS84.

    Args:
        wkt: WKT point string, e.g. "POINT(528349 186246)"
        source_srid: SRID of the input coordinates (4326 or 27700)

    Returns:
        (longitude, latitude) tuple in WGS84 (SRID 4326)

    Raises:
        ValueError: If transformed coordinate is outside UK bounding box
        ValueError: If source_srid is not 4326 or 27700
    """
    inner = wkt[6:-1]  # Strip "POINT(" and ")"
    x, y = map(float, inner.split())

    if source_srid == 27700:
        lon, lat = _BNG_TO_WGS84.transform(x, y)
    elif source_srid == 4326:
        lon, lat = x, y
    else:
        raise ValueError(f"Unsupported SRID: {source_srid}. Use 4326 or 27700.")

    if not (_UK_LAT_MIN <= lat <= _UK_LAT_MAX and _UK_LON_MIN <= lon <= _UK_LON_MAX):
        raise ValueError(
            f"Coordinate ({lon:.4f}, {lat:.4f}) outside UK bounds "
            f"(lat {_UK_LAT_MIN}-{_UK_LAT_MAX}, lon {_UK_LON_MIN}-{_UK_LON_MAX})"
        )

    return lon, lat
