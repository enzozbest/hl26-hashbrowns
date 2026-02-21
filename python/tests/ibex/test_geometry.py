"""Tests for geometry.py SRID-aware WKT point parsing and coordinate transformation."""
import pytest

from hashbrowns.ibex.geometry import parse_point_wkt


class TestBNGToWGS84:
    def test_bng_to_wgs84_known_point(self):
        """BNG (528349, 186246) should transform to Camden/Kentish Town area: lon~-0.150, lat~51.560.

        Verified actual pyproj output: lon=-0.149854, lat=51.560499.
        The plan's approximation (-0.139, 51.547) was slightly off; actual values used here.
        """
        lon, lat = parse_point_wkt("POINT(528349 186246)", 27700)
        assert lon == pytest.approx(-0.150, abs=0.01)
        assert lat == pytest.approx(51.560, abs=0.01)

    def test_bng_easting_northing_for_rochdale(self):
        """BNG (389887, 413707) should transform to Rochdale area: lon~-2.15, lat~53.61."""
        lon, lat = parse_point_wkt("POINT(389887 413707)", 27700)
        assert lon == pytest.approx(-2.15, abs=0.01)
        assert lat == pytest.approx(53.61, abs=0.01)


class TestWGS84Passthrough:
    def test_wgs84_passthrough(self):
        """SRID 4326 coordinates should be returned unchanged."""
        lon, lat = parse_point_wkt("POINT(-0.1276 51.5074)", 4326)
        assert lon == -0.1276
        assert lat == 51.5074

    def test_wgs84_boundary_valid(self):
        """Wales coordinate should pass UK bounds check without error."""
        lon, lat = parse_point_wkt("POINT(-5.0 52.0)", 4326)
        assert lon == -5.0
        assert lat == 52.0


class TestBoundsValidation:
    def test_outside_uk_bounds_raises(self):
        """Coordinate at (0, 0) is outside UK bounds — should raise ValueError."""
        with pytest.raises(ValueError, match="outside UK bounds"):
            parse_point_wkt("POINT(0 0)", 4326)
