"""
pages/04_live_prediction.py
---------------------------
Page 4 — Live Fault Prediction & Explanation

Thesis alignment
----------------
Contributions 3 & 4 (ML + dashboard-based decision support):
This page is the operational prototype — the artefact that a facility
manager would actually use. They enter current sensor readings (or
let sliders auto-populate from the last historical record) and
instantly receive:
  1. A predicted fault class with confidence probabilities
  2. A SHAP waterfall explanation of WHY that prediction was made
  3. A tailored maintenance recommendation

This page is the bridge between "technical model" and "practical tool",
directly satisfying the Design Science Research (DSR) component
(Hevner et al., 2004) of the thesis methodology.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import FAULT_STYLE, LABEL_MAP, RECOMMENDATIONS, SENSOR_META, THESIS_META
from model_utils import compute_shap_single, predict_single
from visualizations import plot_input_vs_historical, plot_prediction_probabilities, plot_shap_waterfall_single


def render(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    explainer,
) -> None:
    """Render the Live Fault Prediction & Explanation page."""

    st.markdown("## ⚡ Live Fault Prediction & Explanation")
    st.markdown(
        "_Enter current sensor readings to receive an instant fault diagnosis "
        "with a SHAP-based explanation. This is **Thesis Contribution 3**: "
        "ML + dashboard-based decision support for facility managers._"
    )
    st.divider()

    source_df = filtered_df if not filtered_df.empty else df
    latest    = source_df.iloc[-1]

    # ── Sensor input panel ────────────────────────────────────────────────────
    st.markdown("### 🎛️  Enter Sensor Readings")
    st.caption(
        "Sliders are pre-populated with the latest historical reading. "
        "Adjust to simulate different operating conditions."
    )

    # Group features into columns for a cleaner layout
    input_data: dict[str, float] = {}
    n_cols = min(3, len(feature_cols))
    cols   = st.columns(n_cols)

    for i, feature in enumerate(feature_cols):
        col = cols[i % n_cols]
        f_min  = float(df[feature].min())
        f_max  = float(df[feature].max())
        f_mean = float(df[feature].mean())
        # Default to latest reading for a realistic starting point
        default = float(latest[feature]) if feature in latest.index else f_mean
        step    = max(round((f_max - f_min) / 200, 4), 0.0001)

        input_data[feature] = col.slider(
            label=feature.replace("_", " "),
            min_value=f_min,
            max_value=f_max,
            value=default,
            step=step,
            key=f"live_{feature}",
        )

    # Reset button
    if st.button("🔄 Reset to Latest Historical Values", use_container_width=False):
        for feature in feature_cols:
            st.session_state[f"live_{feature}"] = float(latest.get(feature, df[feature].mean()))
        st.rerun()

    st.divider()

    # ── Run prediction ────────────────────────────────────────────────────────
    st.markdown("### 🔮 Prediction Results")

    try:
        encoded, class_name, proba_dict = predict_single(model, feature_cols, input_data)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Status banner
    sev_fn = getattr(st, FAULT_STYLE.get(class_name, ("info", ""))[0])
    confidence = proba_dict.get(class_name, 0.0)
    sev_fn(
        f"### Predicted Fault: **{class_name}**  "
        f"_(Confidence: {confidence*100:.1f}%)_"
    )

    # Probability chart
    col_prob, col_rec = st.columns([1, 1])
    with col_prob:
        st.markdown("**Fault Class Probabilities**")
        try:
            fig = plot_prediction_probabilities(proba_dict)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Probability chart error: {e}")

    with col_rec:
        st.markdown("**Maintenance Recommendation**")
        rec = RECOMMENDATIONS.get(class_name, "Review system manually.")
        level = FAULT_STYLE.get(class_name, ("info", ""))[0]
        getattr(st, level)(rec)

        # Severity indicator
        severity_labels = {
            "Healthy":               ("🟢", "Normal"),
            "Bearing Wear":          ("🟡", "Moderate — Schedule within 7 days"),
            "Overheating / Fouling": ("🔴", "High — Immediate action required"),
            "Power Inefficiency":    ("🟡", "Moderate — Investigate within 3 days"),
        }
        icon, label = severity_labels.get(class_name, ("⚪", "Unknown"))
        st.markdown(f"**Severity:** {icon} {label}")

    st.divider()

    # ── SHAP Waterfall (per-prediction explanation) ───────────────────────────
    st.markdown("### 🧠 Why This Prediction? (SHAP Explanation)")
    st.caption(
        "The waterfall chart shows which sensor readings pushed the model toward "
        "or away from the predicted fault class. "
        "🟢 green = increases the fault score; 🔴 red = decreases it. "
        "This is the per-prediction explanation that makes the model **transparent** "
        "and **trustworthy** for facility managers — Contribution 2."
    )
    try:
        with st.spinner("Computing SHAP explanation …"):
            shap_values_single, X_single = compute_shap_single(
                explainer, input_data, feature_cols
            )
        fig = plot_shap_waterfall_single(shap_values_single, X_single, encoded)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP explanation error: {e}")

    st.divider()

    # ── Input vs historical context ───────────────────────────────────────────
    st.markdown("### 📊 Current Input vs Historical Context")
    st.caption(
        "How does the current sensor reading compare to the full historical distribution? "
        "Red dashed line = live input; blue histogram = historical range. "
        "Inputs far from the historical mean are associated with fault events."
    )
    try:
        fig = plot_input_vs_historical(source_df, feature_cols, input_data, max_features=6)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Distribution chart error: {e}")

    st.divider()
    st.caption(
        f"**{THESIS_META['title']}**  ·  {THESIS_META['model']}  ·  "
        f"{THESIS_META['xai']}  ·  Framework: {THESIS_META['framework']}"
    )
