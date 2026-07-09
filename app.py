"""DataQ - Local Data Quality Dashboard.

Entry point for the Streamlit application. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from checks.freshness import check_freshness
from checks.profiling import profile_dataframe
from checks.quality import (
    check_duplicates,
    check_nulls,
    check_ranges,
    check_uniqueness,
)
from checks.schema import detect_drift, save_baseline
from scoring import compute_score
from summarize import summarize_issues, summarize_with_gemini

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"

_GRADE_COLORS = {
    "A": "#2ecc71",
    "B": "#27ae60",
    "C": "#f1c40f",
    "D": "#e67e22",
    "F": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(data))


@st.cache_data(show_spinner=False)
def _read_csv_path(path: str, mtime: float) -> pd.DataFrame:
    # ``mtime`` participates in the cache key so edits invalidate the cache.
    del mtime
    return pd.read_csv(path)


def _list_sample_files() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p for p in SAMPLE_DIR.glob("*.csv") if p.is_file())


# ---------------------------------------------------------------------------
# Analysis (cached on DataFrame identity via hash)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _profile(df: pd.DataFrame) -> pd.DataFrame:
    return profile_dataframe(df)


def _run_all_checks(
    df: pd.DataFrame,
    *,
    null_threshold: float,
    key_columns: list[str],
    range_rules: dict[str, tuple],
    date_column: str | None,
    max_age_days: float,
    baseline_name: str,
) -> list[dict]:
    issues: list[dict] = []
    issues += check_nulls(df, threshold=null_threshold)
    issues += check_duplicates(df)
    if key_columns:
        issues += check_uniqueness(df, key_columns)
    if range_rules:
        issues += check_ranges(df, range_rules)
    issues += detect_drift(df, baseline_name)
    if date_column:
        issues += check_freshness(df, date_column, max_age_days=max_age_days)
    return issues


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _score_gauge(score: int, grade: str) -> go.Figure:
    color = _GRADE_COLORS.get(grade, "#3498db")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 60], "color": "#fdecea"},
                    {"range": [60, 80], "color": "#fff4e5"},
                    {"range": [80, 100], "color": "#e8f8f0"},
                ],
            },
            title={"text": "Data Quality Score"},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def _null_pct_chart(profile_df: pd.DataFrame) -> go.Figure:
    data = profile_df[["column", "null_pct"]].sort_values(
        "null_pct", ascending=False
    )
    fig = px.bar(
        data,
        x="column",
        y="null_pct",
        labels={"null_pct": "Null %", "column": "Column"},
        title="Null percentage by column",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(range=[0, 100])
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="DataQ", layout="wide")
    st.title("DataQ - Local Data Quality Dashboard")

    # ------------------------- Sidebar ------------------------------------
    with st.sidebar:
        st.header("Settings")
        null_threshold = st.slider(
            "Null threshold (flag columns above)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
        max_age_days = st.number_input(
            "Freshness max age (days)",
            min_value=0.0,
            value=1.0,
            step=0.5,
        )
        baseline_name = st.text_input("Baseline name", value="default")
        key_columns_raw = st.text_input(
            "Key columns (comma-separated)", value=""
        )
        date_column = st.text_input("Date column (for freshness)", value="")

    key_columns = [c.strip() for c in key_columns_raw.split(",") if c.strip()]

    # ------------------------- Data loader --------------------------------
    st.subheader("Load data")
    load_col1, load_col2 = st.columns([1, 1])
    with load_col1:
        upload = st.file_uploader("Upload a CSV", type=["csv"])
    with load_col2:
        samples = _list_sample_files()
        sample_labels = ["(none)"] + [p.name for p in samples]
        chosen = st.selectbox(
            "…or pick from sample_data/",
            sample_labels,
            index=0,
        )

    df: pd.DataFrame | None = None
    source_label = ""
    if upload is not None:
        df = _read_csv_bytes(upload.getvalue())
        source_label = f"upload: {upload.name}"
    elif chosen and chosen != "(none)":
        path = SAMPLE_DIR / chosen
        df = _read_csv_path(str(path), path.stat().st_mtime)
        source_label = f"sample_data/{chosen}"

    if df is None:
        st.info("Upload a CSV or pick a sample file to begin.")
        return

    st.caption(f"Loaded **{source_label}** - {len(df):,} rows x {df.shape[1]} cols")

    # ------------------------- Baseline button ----------------------------
    with st.sidebar:
        st.divider()
        if st.button("Save current schema as baseline", use_container_width=True):
            path = save_baseline(df, baseline_name)
            st.success(f"Saved baseline to {path.relative_to(APP_DIR)}")

    # ------------------------- Analysis -----------------------------------
    profile_df = _profile(df)

    # Range rules: users can extend by editing this dict; kept empty by default.
    range_rules: dict[str, tuple] = {}

    issues = _run_all_checks(
        df,
        null_threshold=null_threshold,
        key_columns=key_columns,
        range_rules=range_rules,
        date_column=date_column or None,
        max_age_days=max_age_days,
        baseline_name=baseline_name,
    )
    score_info = compute_score(issues)

    # ------------------------- Top row ------------------------------------
    top1, top2, top3 = st.columns([2, 1, 2])
    with top1:
        st.plotly_chart(
            _score_gauge(score_info["score"], score_info["grade"]),
            use_container_width=True,
        )
    with top2:
        st.metric("Grade", score_info["grade"])
        st.metric("Total issues", len(issues))
    with top3:
        counts = score_info["counts_by_severity"]
        st.metric("High", counts.get("high", 0))
        st.metric("Medium", counts.get("medium", 0))
        st.metric("Low", counts.get("low", 0))

    # ------------------------- Tabs ---------------------------------------
    tab_summary, tab_profile, tab_charts = st.tabs(
        ["Summary", "Profile", "Charts"]
    )

    with tab_summary:
        use_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        if use_gemini:
            summary_text = summarize_with_gemini(issues, score_info)
            summary_source = "Gemini 2.5 Flash"
        else:
            summary_text = summarize_issues(issues, score_info)
            summary_source = "rule-based"
        st.caption(f"Summary source: {summary_source}")
        st.markdown(summary_text)
        st.markdown("---")
        st.markdown("**All issues**")
        if issues:
            st.dataframe(
                pd.DataFrame(issues),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No issues detected.")

    with tab_profile:
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    with tab_charts:
        if profile_df.empty:
            st.info("No columns to chart.")
        else:
            st.plotly_chart(_null_pct_chart(profile_df), use_container_width=True)


if __name__ == "__main__":
    main()
