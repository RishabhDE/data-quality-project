"""Row-level data quality checks.

Each check returns a list of issue dicts with the shape::

    {"check": str, "column": str | None, "severity": str, "message": str}

``severity`` is one of ``"high"``, ``"medium"``, or ``"low"``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd
from pandas.api.types import is_numeric_dtype

Issue = dict


def _severity_for_null_pct(null_pct: float) -> str:
    if null_pct >= 0.5:
        return "high"
    if null_pct >= 0.3:
        return "medium"
    return "low"


def check_nulls(df: pd.DataFrame, threshold: float = 0.2) -> list[Issue]:
    """Flag columns whose null percentage exceeds ``threshold``.

    Args:
        df: DataFrame to inspect.
        threshold: Fraction in ``[0, 1]``. Columns whose null rate is strictly
            greater than this value are flagged.

    Returns:
        A list of issue dicts, one per offending column.
    """
    issues: list[Issue] = []
    if df.shape[1] == 0 or len(df) == 0:
        return issues

    row_count = len(df)
    for name in df.columns:
        null_pct = float(df[name].isna().sum()) / row_count
        if null_pct > threshold:
            issues.append(
                {
                    "check": "nulls",
                    "column": name,
                    "severity": _severity_for_null_pct(null_pct),
                    "message": (
                        f"Column '{name}' is {null_pct:.1%} null "
                        f"(threshold {threshold:.1%})."
                    ),
                }
            )
    return issues


def check_duplicates(df: pd.DataFrame) -> list[Issue]:
    """Flag fully duplicated rows in the DataFrame.

    Args:
        df: DataFrame to inspect.

    Returns:
        A single-element list describing the duplicate-row count, or an
        empty list if none are found.
    """
    if len(df) == 0:
        return []

    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []

    dup_pct = dup_count / len(df)
    severity = "high" if dup_pct >= 0.1 else "medium" if dup_pct >= 0.01 else "low"
    return [
        {
            "check": "duplicates",
            "column": None,
            "severity": severity,
            "message": (
                f"Found {dup_count} fully duplicated row(s) "
                f"({dup_pct:.1%} of rows)."
            ),
        }
    ]


def check_uniqueness(
    df: pd.DataFrame, key_columns: Iterable[str]
) -> list[Issue]:
    """Flag duplicate values in supposed key columns.

    Each column in ``key_columns`` is checked individually. Missing columns
    are reported as a ``high``-severity issue.

    Args:
        df: DataFrame to inspect.
        key_columns: Column names expected to be unique.

    Returns:
        A list of issue dicts.
    """
    issues: list[Issue] = []
    for col in key_columns:
        if col not in df.columns:
            issues.append(
                {
                    "check": "uniqueness",
                    "column": col,
                    "severity": "high",
                    "message": f"Key column '{col}' is missing from the DataFrame.",
                }
            )
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        dup_count = int(series.duplicated().sum())
        if dup_count > 0:
            issues.append(
                {
                    "check": "uniqueness",
                    "column": col,
                    "severity": "high",
                    "message": (
                        f"Key column '{col}' has {dup_count} duplicate value(s)."
                    ),
                }
            )
    return issues


def check_ranges(
    df: pd.DataFrame, rules: Mapping[str, tuple]
) -> list[Issue]:
    """Flag values outside of ``[min, max]`` for each configured column.

    ``rules`` maps a column name to a ``(min, max)`` tuple. Either bound may
    be ``None`` to leave that side open. Missing columns and non-numeric
    columns are reported as issues so the caller can correct the config.

    Args:
        df: DataFrame to inspect.
        rules: Mapping of column name -> ``(min, max)``.

    Returns:
        A list of issue dicts.
    """
    issues: list[Issue] = []
    for col, bounds in rules.items():
        if col not in df.columns:
            issues.append(
                {
                    "check": "ranges",
                    "column": col,
                    "severity": "high",
                    "message": f"Range rule references missing column '{col}'.",
                }
            )
            continue

        series = df[col]
        if not is_numeric_dtype(series):
            issues.append(
                {
                    "check": "ranges",
                    "column": col,
                    "severity": "medium",
                    "message": (
                        f"Range rule on non-numeric column '{col}' "
                        f"(dtype {series.dtype}) was skipped."
                    ),
                }
            )
            continue

        lo, hi = bounds
        mask = pd.Series(False, index=series.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        mask &= series.notna()

        offending = int(mask.sum())
        if offending == 0:
            continue

        pct = offending / len(series) if len(series) else 0.0
        severity = "high" if pct >= 0.1 else "medium" if pct >= 0.01 else "low"
        lo_txt = "-inf" if lo is None else lo
        hi_txt = "+inf" if hi is None else hi
        issues.append(
            {
                "check": "ranges",
                "column": col,
                "severity": severity,
                "message": (
                    f"Column '{col}' has {offending} value(s) outside "
                    f"[{lo_txt}, {hi_txt}] ({pct:.1%} of rows)."
                ),
            }
        )
    return issues
