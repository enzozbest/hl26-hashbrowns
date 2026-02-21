"""Data freshness tracking for council-level Ibex API pulls (DATA-01)."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

_DEFAULT_PATH = Path(".ibex_freshness.json")


def record_pull(council_id: int, path: Path = _DEFAULT_PATH) -> None:
    """Record that council_id's data was pulled right now.

    Args:
        council_id: Numeric council identifier
        path: Path to the JSON freshness file (default: .ibex_freshness.json in cwd)
    """
    data = _load(path)
    data[str(council_id)] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2))


def last_pulled(council_id: int, path: Path = _DEFAULT_PATH) -> datetime | None:
    """Return the datetime of the last pull for council_id, or None if never pulled.

    Args:
        council_id: Numeric council identifier
        path: Path to the JSON freshness file (default: .ibex_freshness.json in cwd)

    Returns:
        Timezone-aware UTC datetime of last pull, or None if no record exists
    """
    data = _load(path)
    ts = data.get(str(council_id))
    if ts is None:
        return None
    return datetime.fromisoformat(ts)


def is_stale(
    council_id: int, max_age_days: int = 30, path: Path = _DEFAULT_PATH
) -> bool:
    """Return True if council_id's data has not been pulled or is older than max_age_days.

    Args:
        council_id: Numeric council identifier
        max_age_days: Maximum acceptable age of data in days (default: 30)
        path: Path to the JSON freshness file (default: .ibex_freshness.json in cwd)

    Returns:
        True if data is stale or never pulled; False if within max_age_days
    """
    pulled = last_pulled(council_id, path)
    if pulled is None:
        return True
    age = datetime.now(timezone.utc) - pulled
    return age.days >= max_age_days


def _load(path: Path) -> dict:
    """Load the freshness JSON file, returning an empty dict if it doesn't exist."""
    if path.exists():
        return json.loads(path.read_text())
    return {}
