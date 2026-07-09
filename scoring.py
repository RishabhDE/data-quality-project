"""Data quality scoring.

Aggregate individual check issue dicts into a weighted overall data quality
score (0-100) with a letter grade and per-severity counts.
"""

from __future__ import annotations

from typing import Iterable

_SEVERITY_WEIGHTS: dict[str, int] = {
    "high": 15,
    "medium": 7,
    "low": 3,
}


def _grade_for(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_score(issues: Iterable[dict]) -> dict:
    """Compute an overall data quality score from a list of issues.

    Starts at 100 and subtracts weighted points per issue based on severity
    (``high=15``, ``medium=7``, ``low=3``). The score is floored at 0 and
    mapped to a letter grade (A >= 90, B >= 80, C >= 70, D >= 60, else F).

    Args:
        issues: Iterable of issue dicts, each containing a ``"severity"`` key
            with one of ``"high"``, ``"medium"``, or ``"low"``. Unknown
            severities are counted but do not deduct points.

    Returns:
        A dict with keys:
            * ``score`` (int, 0-100)
            * ``grade`` (str, one of ``"A"``-``"F"``)
            * ``counts_by_severity`` (dict[str, int])
    """
    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    deduction = 0

    for issue in issues:
        severity = str(issue.get("severity", "")).lower()
        counts[severity] = counts.get(severity, 0) + 1
        deduction += _SEVERITY_WEIGHTS.get(severity, 0)

    score = max(0, 100 - deduction)
    return {
        "score": score,
        "grade": _grade_for(score),
        "counts_by_severity": counts,
    }
