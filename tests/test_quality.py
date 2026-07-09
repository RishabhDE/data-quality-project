"""Tests for the quality checks and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable when running pytest from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks.quality import check_duplicates, check_nulls  # noqa: E402
from scoring import compute_score  # noqa: E402


# ---------------------------------------------------------------------------
# check_nulls
# ---------------------------------------------------------------------------


def test_check_nulls_flags_columns_above_threshold():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],           # 0% null
            "b": [1, np.nan, np.nan, 4, 5], # 40% null
            "c": [np.nan] * 5,              # 100% null
        }
    )

    issues = check_nulls(df, threshold=0.2)

    flagged = {issue["column"] for issue in issues}
    assert flagged == {"b", "c"}
    assert all(issue["check"] == "nulls" for issue in issues)


def test_check_nulls_severity_scales_with_null_pct():
    df = pd.DataFrame(
        {
            "low": [1, 2, 3, np.nan, np.nan],       # 40% -> medium
            "high": [np.nan] * 4 + [1],             # 80% -> high
        }
    )

    issues = check_nulls(df, threshold=0.2)
    by_col = {i["column"]: i["severity"] for i in issues}

    assert by_col["low"] == "medium"
    assert by_col["high"] == "high"


def test_check_nulls_respects_threshold():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4, 5]})  # 20% null

    # Threshold is strictly greater than, so exactly 20% at threshold=0.2 passes.
    assert check_nulls(df, threshold=0.2) == []
    # A lower threshold flags it.
    assert len(check_nulls(df, threshold=0.1)) == 1


def test_check_nulls_empty_inputs():
    assert check_nulls(pd.DataFrame()) == []
    assert check_nulls(pd.DataFrame({"a": []})) == []


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------


def test_check_duplicates_returns_empty_for_unique_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert check_duplicates(df) == []


def test_check_duplicates_reports_duplicate_count():
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3, 3, 3], "b": ["x", "x", "y", "z", "z", "z"]}
    )

    issues = check_duplicates(df)

    assert len(issues) == 1
    issue = issues[0]
    assert issue["check"] == "duplicates"
    assert issue["column"] is None
    # Rows 1, 4, 5 are duplicates of earlier rows -> 3 duplicates.
    assert "3" in issue["message"]
    assert issue["severity"] in {"low", "medium", "high"}


def test_check_duplicates_empty_frame():
    assert check_duplicates(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------


def test_compute_score_perfect_when_no_issues():
    result = compute_score([])
    assert result == {
        "score": 100,
        "grade": "A",
        "counts_by_severity": {"high": 0, "medium": 0, "low": 0},
    }


def test_compute_score_weights_severity():
    issues = [
        {"severity": "high", "check": "x", "column": None, "message": ""},
        {"severity": "medium", "check": "x", "column": None, "message": ""},
        {"severity": "low", "check": "x", "column": None, "message": ""},
    ]
    result = compute_score(issues)

    # 100 - 15 - 7 - 3 = 75 -> grade C
    assert result["score"] == 75
    assert result["grade"] == "C"
    assert result["counts_by_severity"] == {"high": 1, "medium": 1, "low": 1}


def test_compute_score_floors_at_zero():
    issues = [{"severity": "high"} for _ in range(20)]
    result = compute_score(issues)
    assert result["score"] == 0
    assert result["grade"] == "F"
    assert result["counts_by_severity"]["high"] == 20


@pytest.mark.parametrize(
    "issues",
    [
        [],
        [{"severity": "high"}],
        [{"severity": "high"}, {"severity": "medium"}],
        [{"severity": "high"}, {"severity": "high"}],
        [{"severity": "medium"}] * 6,
        [{"severity": "low"}] * 4,
    ],
)
def test_compute_score_matches_weighted_formula(issues):
    weights = {"high": 15, "medium": 7, "low": 3}
    expected_score = max(0, 100 - sum(weights[i["severity"]] for i in issues))
    if expected_score >= 90:
        expected_grade = "A"
    elif expected_score >= 80:
        expected_grade = "B"
    elif expected_score >= 70:
        expected_grade = "C"
    elif expected_score >= 60:
        expected_grade = "D"
    else:
        expected_grade = "F"

    result = compute_score(issues)
    assert result["score"] == expected_score
    assert result["grade"] == expected_grade
