"""
app.py
------
HVAC Chiller Predictive Maintenance Dashboard
==============================================

Entry point for the Streamlit application.

Run with:
    streamlit run app.py

Project structure
-----------------
app.py                  ← this file (routing + sidebar)
config.py               ← all constants, paths, labels
data_loader.py          ← cached I/O and data processing
model_utils.py          ← inference and SHAP computations
visualizations.py       ← all chart rendering functions
pages/
    01_overview.py      ← System Overview (Contribution 4)
    02_fault_analysis.py← Fault Classification (Contribution 1)
    03_explainability.py← SHAP Explainability (Contribution 2)
    04_live_prediction.py← Live Prediction Tool (Contributions 3 & 4)

Thesis
------
Title:     Explainable ML for HVAC Chiller Predictive Maintenance
Model:     Random Forest Classifier (scikit-learn)
XAI:       SHAP TreeExplainer (Lundberg & Lee, 2017)
Framework: Design Science Research (Hevner et al., 2004)
"""

import streamlit as st

# ── Streamlit page config (must be first st call) ────────────────────────────
st.set_page_config(
    page_title="HVAC Predictive Maintenance",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import LABEL_MAP, THESIS_META
from data_loader import (
    build_explainer,
    check_required_files,
    filter_dataframe,
    load_data,
    load_feature_cols,
    load_model,
    validate_columns,
)

# ── Required file check ───────────────────────────────────────────────────────
missing = check_required_files()
if missing:
    st.error("⚠️  Missing required files. Please run the data preparation pipeline first.")
    for f in missing:
        st.code(f)
    st.stop()

# ── Load resources ────────────────────────────────────────────────────────────
try:
    df           = load_data()
    model        = load_model()
    feature_cols = load_feature_cols()
    explainer    = build_explainer(model)
except Exception as exc:
    st.error(f"Failed to load resources: {exc}")
    st.stop()

# ── Column validation ─────────────────────────────────────────────────────────
bad_cols = validate_columns(df, feature_cols)
if bad_cols:
    st.error(f"Data is missing expected columns: {bad_cols}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-conditioner.png", width=64)
    st.markdown(f"## {THESIS_META['subtitle']}")
    st.caption(THESIS_META["title"])
    st.divider()

    # Page navigation
    st.markdown("### 📂 Navigation")
    page = st.radio(
        "Select page",
        options=[
            "🏢  System Overview",
            "🔍  Fault Analysis",
            "🧠  Explainability (SHAP)",
            "⚡  Live Prediction",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Date filter ──────────────────────────────────────────────────────────
    st.markdown("### 📅 Date Filter")
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()

    c1, c2 = st.columns(2)
    start_date = c1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    end_date   = c2.date_input("To",   value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before end date.")

    # ── Fault type filter ─────────────────────────────────────────────────────
    st.markdown("### 🏷️  Fault Type Filter")
    all_faults     = sorted(df["Predicted_Class_Name"].dropna().unique().tolist())
    selected_faults = st.multiselect(
        "Include fault classes",
        options=all_faults,
        default=all_faults,
        label_visibility="collapsed",
    )

    st.divider()

    # Data summary
    st.markdown("### 📊 Dataset Info")
    st.caption(f"Total records: **{len(df):,}**")
    st.caption(f"Date range: **{min_date}** → **{max_date}**")
    st.caption(f"Features: **{len(feature_cols)}**")
    st.caption(f"Fault classes: **{len(LABEL_MAP)}**")

    st.divider()
    st.caption(
        f"Model: {THESIS_META['model']}\n\n"
        f"XAI: {THESIS_META['xai']}\n\n"
        f"Framework: {THESIS_META['framework']}"
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered_df = filter_dataframe(df, start_date, end_date, selected_faults)

# ── Page routing ──────────────────────────────────────────────────────────────
if page.startswith("🏢"):
    from pages.overview import render
    render(df, filtered_df)

elif page.startswith("🔍"):
    from pages.fault_analysis import render
    render(df, filtered_df)

elif page.startswith("🧠"):
    from pages.explainability import render
    render(df, filtered_df, model, feature_cols, explainer)

elif page.startswith("⚡"):
    from pages.live_prediction import render
    render(df, filtered_df, model, feature_cols, explainer)
