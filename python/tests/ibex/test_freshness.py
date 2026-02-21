"""Tests for freshness.py per-council data freshness tracking."""
import json
from datetime import datetime, timezone, timedelta

import pytest

from hashbrowns.ibex.freshness import record_pull, last_pulled, is_stale


class TestRecordAndRetrieve:
    def test_record_and_retrieve(self, tmp_path):
        """record_pull followed by last_pulled should return a datetime."""
        path = tmp_path / "fresh.json"
        record_pull(240, path=path)
        result = last_pulled(240, path=path)
        assert result is not None
        assert isinstance(result, datetime)

    def test_last_pulled_missing_council(self, tmp_path):
        """last_pulled for a council never recorded should return None."""
        path = tmp_path / "fresh.json"
        result = last_pulled(999, path=path)
        assert result is None


class TestStaleness:
    def test_never_pulled_is_stale(self, tmp_path):
        """Council with no pull record should be stale."""
        path = tmp_path / "fresh.json"
        assert is_stale(240, path=path) is True

    def test_recently_pulled_not_stale(self, tmp_path):
        """Council pulled just now should not be stale within 30 days."""
        path = tmp_path / "fresh.json"
        record_pull(240, path=path)
        assert is_stale(240, max_age_days=30, path=path) is False

    def test_old_pull_is_stale(self, tmp_path):
        """Council pulled 31 days ago should be stale when max_age_days=30."""
        path = tmp_path / "fresh.json"
        old_dt = datetime.now(timezone.utc) - timedelta(days=31)
        path.write_text(json.dumps({"240": old_dt.isoformat()}))
        assert is_stale(240, max_age_days=30, path=path) is True


class TestMultipleCouncils:
    def test_multiple_councils_independent(self, tmp_path):
        """Staleness for council 240 and 241 should be tracked independently."""
        path = tmp_path / "fresh.json"
        # Neither recorded yet
        assert is_stale(240, path=path) is True
        assert is_stale(241, path=path) is True

        # Record council 240 only
        record_pull(240, path=path)
        assert is_stale(240, max_age_days=30, path=path) is False
        assert is_stale(241, path=path) is True

        # Record council 241
        record_pull(241, path=path)
        assert is_stale(241, max_age_days=30, path=path) is False
