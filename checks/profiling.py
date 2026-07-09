"""Dataset profiling.

Compute descriptive statistics (row counts, dtypes, null rates, cardinality,
numeric distributions) for an input DataFrame.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

_PROFILE_COLUMNS = [
    "column",
    "dtype",
    "non_null_count",
    "null_count",
    "null_pct",
    "distinct_count",
    "min",
    "max",
    "mean",
]


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column profile of ``df``.

    Produces one row per column with dtype, null counts, null percentage,
    distinct count, and (for numeric columns) min, max, and mean. Non-numeric
    columns have ``NaN`` in the numeric summary fields. An empty input (no
    columns) yields an empty profile with the expected schema.

    Args:
        df: The DataFrame to profile.

    Returns:
        A DataFrame with columns: ``column``, ``dtype``, ``non_null_count``,
        ``null_count``, ``null_pct``, ``distinct_count``, ``min``, ``max``,
        ``mean``.
    """
    if df.shape[1] == 0:
        return pd.DataFrame(columns=_PROFILE_COLUMNS)

    row_count = len(df)
    rows: list[dict] = []
    for name in df.columns:
        col = df[name]
        non_null = int(col.notna().sum())
        null = int(col.isna().sum())
        null_pct = (null / row_count * 100.0) if row_count else 0.0

        row: dict = {
            "column": name,
            "dtype": str(col.dtype),
            "non_null_count": non_null,
            "null_count": null,
            "null_pct": null_pct,
            "distinct_count": int(col.nunique(dropna=True)),
            "min": pd.NA,
            "max": pd.NA,
            "mean": pd.NA,
        }

        if is_numeric_dtype(col) and non_null > 0:
            row["min"] = col.min()
            row["max"] = col.max()
            row["mean"] = float(col.mean())

        rows.append(row)

    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
