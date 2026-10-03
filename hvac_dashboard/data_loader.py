"""
data_loader.py
--------------
Handles all data loading, validation, and preprocessing.
Every function that touches the filesystem lives here.
Uses Streamlit's caching decorators so expensive I/O runs
only once per session.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    DATA_PATH,
    FEATURE_COLS_PATH,
    LABEL_MAP,
    MODEL_PATH,
    PROB_COL_MAP,
    REQUIRED_FILES,
)


# ── File validation ──────────────────────────────────────────────────────────

def check_required_files() -> list[str]:
    """Return a list of missing required file paths (as strings)."""
    return [str(p) for p in REQUIRED_FILES if not p.exists()]


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading prediction data …")
def load_data() -> pd.DataFrame:
    """
    Load and preprocess the predictions CSV.

    Adds human-readable class name columns and ensures the
    Timestamp column is parsed as datetime.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for the dashboard.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

    # Human-readable class labels
    df["Predicted_Class_Name"] = df["Predicted_Label"].map(LABEL_MAP)
    df["Actual_Class_Name"]    = df["Actual_Label"].map(LABEL_MAP)

    # Ensure probability columns are numeric
    for col in PROB_COL_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.sort_values("Timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """
    Load the trained Random Forest model from disk.

    Returns
    -------
    sklearn estimator
        The fitted RandomForestClassifier.
    """
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading feature list …")
def load_feature_cols() -> list[str]:
    """
    Load the ordered list of feature column names used during training.

    Returns
    -------
    list[str]
        Feature column names in training order.
    """
    import joblib
    return joblib.load(FEATURE_COLS_PATH)


@st.cache_resource(show_spinner="Building SHAP explainer …")
def build_explainer(_model):
    """
    Build and cache a SHAP TreeExplainer for the given model.

    The leading underscore in ``_model`` prevents Streamlit from
    attempting to hash the sklearn estimator (which is not hashable).

    Parameters
    ----------
    _model : sklearn estimator
        The trained RandomForestClassifier.

    Returns
    -------
    shap.TreeExplainer
    """
    import shap
    return shap.TreeExplainer(_model)


# ── Column validation ────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """
    Return a list of column names that are expected but absent from *df*.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    list[str]
        Missing column names (empty list if all present).
    """
    expected = list(PROB_COL_MAP.values()) + feature_cols
    return [c for c in expected if c not in df.columns]


# ── Filtering helpers ────────────────────────────────────────────────────────

def filter_dataframe(
    df: pd.DataFrame,
    start_date,
    end_date,
    selected_faults: list[str],
) -> pd.DataFrame:
    """
    Apply date range and fault type filters to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    start_date : datetime.date
    end_date : datetime.date
    selected_faults : list[str]
        Fault class names to include.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    mask = (
        (df["Timestamp"].dt.date >= start_date)
        & (df["Timestamp"].dt.date <= end_date)
        & (df["Predicted_Class_Name"].isin(selected_faults))
    )
    return df[mask].reset_index(drop=True)


# ── Summary statistics ───────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute headline KPI statistics for the overview page.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe.

    Returns
    -------
    dict with keys:
        total_records, fault_rate_pct, most_common_fault,
        healthy_pct, alert_count
    """
    if df.empty:
        return {
            "total_records":    0,
            "fault_rate_pct":   0.0,
            "most_common_fault": "N/A",
            "healthy_pct":      0.0,
            "alert_count":      0,
        }

    total    = len(df)
    healthy  = (df["Predicted_Class_Name"] == "Healthy").sum()
    fault_ct = total - healthy

    alert_classes = {"Overheating / Fouling", "Bearing Wear", "Power Inefficiency"}
    alert_count   = df["Predicted_Class_Name"].isin(alert_classes).sum()

    most_common = (
        df[df["Predicted_Class_Name"] != "Healthy"]["Predicted_Class_Name"]
        .value_counts()
        .idxmax()
        if fault_ct > 0
        else "None"
    )

    return {
        "total_records":     total,
        "fault_rate_pct":    round(fault_ct / total * 100, 1),
        "most_common_fault": most_common,
        "healthy_pct":       round(healthy / total * 100, 1),
        "alert_count":       int(alert_count),
    }
