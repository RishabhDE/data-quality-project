"""Schema baseline persistence and drift detection.

A baseline is a JSON file mapping ``column name -> dtype string`` stored under
``baselines/<name>.json``. ``detect_drift`` compares an incoming DataFrame's
schema against the stored baseline and reports added columns, removed
columns, and dtype changes as issue dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"

Issue = dict


# Aliases so schemas saved under one pandas version still compare correctly
# when read back on another. Pandas may label text columns as ``object``,
# ``str``, ``string``, ``string[python]``, or ``string[pyarrow]`` depending
# on version and options; they are all "string-y" for our purposes.
_DTYPE_ALIASES: dict[str, str] = {
    "object": "string",
    "str": "string",
    "string": "string",
    "string[python]": "string",
    "string[pyarrow]": "string",
}


def _normalize_dtype(dtype_str: str) -> str:
    """Collapse equivalent dtype spellings to a canonical form."""
    key = dtype_str.strip().lower()
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    # Also collapse pyarrow-backed variants like "string[pyarrow]" that
    # arrive with mixed case.
    if key.startswith("string["):
        return "string"
    return dtype_str


def _schema_from_df(df: pd.DataFrame) -> dict[str, str]:
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def _baseline_path(name: str) -> Path:
    return BASELINES_DIR / f"{name}.json"


def save_baseline(df: pd.DataFrame, name: str) -> Path:
    """Persist the schema of ``df`` as ``baselines/<name>.json``.

    Args:
        df: DataFrame whose ``{column: dtype}`` schema will be saved.
        name: Baseline identifier (used as the filename stem).

    Returns:
        The path to the written baseline file.
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = _baseline_path(name)
    schema = _schema_from_df(df)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True))
    return path


def detect_drift(df: pd.DataFrame, name: str) -> list[Issue]:
    """Compare ``df``'s schema to the ``name`` baseline and return issues.

    Reports:
        * ``schema.added`` - column present in ``df`` but not the baseline.
        * ``schema.removed`` - column present in the baseline but not ``df``.
        * ``schema.dtype_changed`` - column whose dtype differs.

    If no baseline exists yet, one is created from ``df`` and a single
    informational note is returned in place of drift issues.

    Args:
        df: DataFrame to check.
        name: Baseline identifier previously used with :func:`save_baseline`.

    Returns:
        A list of issue dicts with keys ``check``, ``column``, ``severity``,
        and ``message``.
    """
    path = _baseline_path(name)
    if not path.exists():
        save_baseline(df, name)
        return [
            {
                "check": "schema.baseline_created",
                "column": None,
                "severity": "low",
                "message": (
                    f"No baseline '{name}' found; created a new baseline at "
                    f"{path}."
                ),
            }
        ]

    baseline: dict[str, str] = json.loads(path.read_text())
    current = _schema_from_df(df)

    issues: list[Issue] = []

    for col in current.keys() - baseline.keys():
        issues.append(
            {
                "check": "schema.added",
                "column": col,
                "severity": "medium",
                "message": (
                    f"Column '{col}' (dtype {current[col]}) is not in "
                    f"baseline '{name}'."
                ),
            }
        )

    for col in baseline.keys() - current.keys():
        issues.append(
            {
                "check": "schema.removed",
                "column": col,
                "severity": "high",
                "message": (
                    f"Column '{col}' from baseline '{name}' is missing "
                    f"(expected dtype {baseline[col]})."
                ),
            }
        )

    for col in baseline.keys() & current.keys():
        if _normalize_dtype(baseline[col]) != _normalize_dtype(current[col]):
            issues.append(
                {
                    "check": "schema.dtype_changed",
                    "column": col,
                    "severity": "high",
                    "message": (
                        f"Column '{col}' dtype changed: "
                        f"{baseline[col]} -> {current[col]}."
                    ),
                }
            )

    return issues
