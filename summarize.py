"""Human-readable summaries of data quality results.

Turn structured issue lists and scores into a concise, plain-English summary
suitable for display in the Streamlit dashboard. A rule-based summary is
always available; an optional Gemini-powered summary is used when the
``GEMINI_API_KEY`` environment variable is set.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

_RECOMMENDATIONS: dict[str, str] = {
    "nulls": "Investigate the upstream source and add null handling or imputation.",
    "duplicates": "Deduplicate on the natural key or fix the ingestion join.",
    "uniqueness": "Enforce a unique constraint on the key column.",
    "ranges": "Add a validation rule or clip outliers before downstream use.",
    "schema.added": "Confirm the new column is expected and update the baseline.",
    "schema.removed": "Restore the missing column or update the baseline if intentional.",
    "schema.dtype_changed": "Cast the column back to its expected dtype at load time.",
    "schema.baseline_created": "Review the auto-created baseline before the next run.",
    "freshness": "Trigger a refresh of the upstream pipeline.",
}


def _recommend(check: str) -> str:
    return _RECOMMENDATIONS.get(check, "Review the issue and address at the source.")


def summarize_issues(issues: Iterable[dict], score_info: dict) -> str:
    """Return a concise plain-English summary of issues and score.

    Includes the overall grade and score, the three most severe issues, and
    a recommended next action for each. Kept under ~120 words.

    Args:
        issues: Iterable of issue dicts (``check``, ``column``, ``severity``,
            ``message``).
        score_info: Output of :func:`scoring.compute_score`.

    Returns:
        A markdown-formatted summary string.
    """
    grade = score_info.get("grade", "?")
    score = score_info.get("score", "?")
    counts = score_info.get("counts_by_severity", {})

    issue_list = list(issues)
    total = len(issue_list)

    header = (
        f"**Overall grade: {grade}** (score {score}/100). "
        f"{total} issue(s): "
        f"{counts.get('high', 0)} high, "
        f"{counts.get('medium', 0)} medium, "
        f"{counts.get('low', 0)} low."
    )

    if total == 0:
        return header + "\n\nNo action needed."

    top = sorted(
        issue_list,
        key=lambda i: _SEVERITY_RANK.get(str(i.get("severity", "")).lower(), 99),
    )[:3]

    lines = [header, "", "**Top issues:**"]
    for issue in top:
        check = str(issue.get("check", "unknown"))
        column = issue.get("column")
        severity = str(issue.get("severity", "?")).lower()
        message = str(issue.get("message", "")).rstrip(".")
        target = f" [{column}]" if column else ""
        lines.append(
            f"- ({severity}) {check}{target}: {message}. "
            f"Next: {_recommend(check)}"
        )

    return "\n".join(lines)


def summarize_with_gemini(issues: Iterable[dict], score_info: dict) -> str:
    """Return a Gemini-generated summary, falling back to the rule-based one.

    If ``GEMINI_API_KEY`` is not set in the environment, or any error occurs
    while calling the API (network, quota, auth, import), this function
    silently returns :func:`summarize_issues` output so the dashboard never
    crashes.

    Args:
        issues: Iterable of issue dicts.
        score_info: Output of :func:`scoring.compute_score`.

    Returns:
        A plain-English summary string (under ~120 words).
    """
    issue_list = list(issues)
    fallback = summarize_issues(issue_list, score_info)

    if not os.environ.get("GEMINI_API_KEY"):
        return fallback

    try:
        from google import genai  # type: ignore

        client = genai.Client()

        payload = {
            "score": score_info.get("score"),
            "grade": score_info.get("grade"),
            "counts_by_severity": score_info.get("counts_by_severity", {}),
            "issues": issue_list,
        }
        prompt = (
            "You are a data quality analyst. Summarize the following data "
            "quality report for a business audience in under 120 words. "
            "Start with the overall grade and score. Then call out the top "
            "3 most severe issues and give exactly one concrete recommended "
            "action per issue. Use plain English, no jargon, no code. If "
            "there are no issues, say so and stop.\n\n"
            "Report (JSON):\n"
            f"{json.dumps(payload, default=str, indent=2)}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (getattr(response, "text", "") or "").strip()
        return text if text else fallback
    except Exception:
        return fallback
