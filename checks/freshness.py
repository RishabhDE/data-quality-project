"""Freshness checks.

Assess how recent the data is based on a timestamp column and a configured
staleness threshold.
"""

from __future__ import annotations

import pandas as pd

Issue = dict


def check_freshness(
    df: pd.DataFrame,
    date_column: str,
    max_age_days: float = 1,
) -> list[Issue]:
    """Flag the DataFrame if its newest ``date_column`` value is stale.

    Parses ``date_column`` to datetimes, finds the maximum, and compares its
    age (in days, from ``pandas.Timestamp.today()``) against ``max_age_days``.
    A single ``high``-severity issue is returned when the data is too old.

    Silently skips (returns ``[]``) when:
        * ``date_column`` is missing from ``df``,
        * the DataFrame is empty,
        * the column cannot be parsed as datetime, or
        * every value is null after parsing.

    Args:
        df: DataFrame to inspect.
        date_column: Name of the timestamp column.
        max_age_days: Maximum allowed age of the most recent record, in days.

    Returns:
        A list containing at most one issue dict.
    """
    if date_column not in df.columns or len(df) == 0:
        return []

    try:
        parsed = pd.to_datetime(df[date_column], errors="coerce", utc=False)
    except (TypeError, ValueError):
        return []

    parsed = parsed.dropna()
    if parsed.empty:
        return []

    latest = parsed.max()
    # Compare naive-to-naive to avoid tz arithmetic surprises.
    now = pd.Timestamp.today()
    if getattr(latest, "tzinfo", None) is not None:
        latest = latest.tz_convert(None) if latest.tz is not None else latest
        try:
            latest = latest.tz_localize(None)
        except (TypeError, AttributeError):
            pass

    age_days = (now - latest).total_seconds() / 86400.0
    if age_days <= max_age_days:
        return []

    return [
        {
            "check": "freshness",
            "column": date_column,
            "severity": "high",
            "message": (
                f"Latest value in '{date_column}' is {age_days:.1f} day(s) old "
                f"(max allowed {max_age_days}); most recent timestamp: {latest}."
            ),
        }
    ]
