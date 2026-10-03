"""
pages/03_explainability.py
--------------------------
Page 3 — SHAP Explainability

Thesis alignment
----------------
Contribution 2 (Integration of explainable AI using SHAP):
This is the academic centrepiece. Every chart on this page directly
demonstrates that the model is not a black box — it provides
interpretable feature attributions that domain experts (facility
managers) can reason about and trust.

The thesis hypothesis is:
  "Vibration features → mechanical faults (Bearing Wear)"
  "Temperature features → thermal faults (Overheating/Fouling)"
  "Power features → efficiency faults (Power Inefficiency)"

The feature–fault attribution heatmap is designed to confirm or
refute these hypotheses visually — it is the key exhibit for §5
(Anticipated Findings) of the thesis proposal.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import LABEL_MAP, SHAP_SAMPLE_SIZE, THESIS_META
from model_utils import (
    build_explainer,
    compute_shap_values,
    mean_abs_shap_per_class,
    overall_feature_importance,
)
from visualizations import (
    plot_shap_beeswarm,
    plot_shap_heatmap,
    plot_shap_summary_bar,
)


def render(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    explainer,
) -> None:
    """Render the SHAP Explainability page."""

    st.markdown("## 🧠 Explainability Analysis (SHAP)")
    st.markdown(
        "_This page demonstrates **Thesis Contribution 2**: integration of SHAP "
        "(SHapley Additive exPlanations) to produce interpretable, per-feature "
        "explanations for every fault prediction. "
        "Reference: Lundberg & Lee (2017) — primary conversant ★._"
    )
    with st.expander("ℹ️  What is SHAP?", expanded=False):
        st.markdown(
            """
**SHAP** assigns each feature a *contribution value* for a given prediction.

- A **positive SHAP value** pushes the model toward predicting this class.
- A **negative SHAP value** pushes the model away from this class.
- **Mean |SHAP|** across many predictions gives a stable importance ranking.

Unlike traditional feature importances (which are global and class-agnostic),
SHAP values are **local** (per prediction) and **class-specific**, making them
ideal for multi-fault classification in HVAC systems.
            """
        )
    st.divider()

    source_df = filtered_df if not filtered_df.empty else df
    if len(source_df) < 5:
        st.warning("Too few records for SHAP analysis. Broaden filters.")
        return

    # ── Compute SHAP (cached via explainer built in main app) ─────────────────
    sample_n = min(len(source_df), SHAP_SAMPLE_SIZE)
    st.info(
        f"SHAP computed on a random sample of **{sample_n}** records "
        f"(full dataset: {len(source_df):,} records) for performance. "
        "Results are representative of the full distribution."
    )

    try:
        with st.spinner("Computing SHAP values …"):
            shap_values, X_sample = compute_shap_values(
                explainer, source_df, feature_cols, sample_size=sample_n
            )
        mean_abs_df = mean_abs_shap_per_class(shap_values, feature_cols)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return

    # ── Overall feature importance ────────────────────────────────────────────
    st.markdown("### 1️⃣  Global Feature Importance (All Classes)")
    st.caption(
        "Mean absolute SHAP value across all fault classes and all samples. "
        "Higher = more influential in driving any fault prediction."
    )
    overall_imp = overall_feature_importance(shap_values, feature_cols)
    imp_df = overall_imp.reset_index()
    imp_df.columns = ["Feature", "Mean |SHAP|"]
    st.bar_chart(imp_df.set_index("Feature"))

    st.divider()

    # ── Multi-class SHAP summary bar ──────────────────────────────────────────
    st.markdown("### 2️⃣  Per-Class SHAP Feature Importance")
    st.caption(
        "Each colour represents one fault class. "
        "Bars show how much each feature contributes **to that specific fault**. "
        "This is the key thesis exhibit: vibration features should dominate "
        "Bearing Wear; temperature features should dominate Overheating/Fouling."
    )
    try:
        fig = plot_shap_summary_bar(shap_values, X_sample)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP summary bar error: {e}")

    st.divider()

    # ── Feature–Fault Attribution Heatmap ────────────────────────────────────
    st.markdown("### 3️⃣  Feature–Fault Attribution Heatmap")
    st.caption(
        "**Primary thesis exhibit.** Each cell = mean |SHAP| for that feature × fault class pair. "
        "Darker = stronger attribution. "
        "Confirms or refutes the thesis hypothesis about which sensor types "
        "drive which fault classes (§5: Anticipated Findings)."
    )
    try:
        fig = plot_shap_heatmap(mean_abs_df)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Heatmap error: {e}")

    # Automated interpretation
    st.markdown("#### 🔬 Automated Interpretation")
    interpretation_rows = []
    for cls in LABEL_MAP.values():
        if cls in mean_abs_df.columns:
            top_feat = mean_abs_df[cls].idxmax()
            top_val  = mean_abs_df[cls].max()
            interpretation_rows.append(
                f"- **{cls}**: most strongly driven by `{top_feat}` "
                f"(mean |SHAP| = {top_val:.4f})"
            )
    st.markdown("\n".join(interpretation_rows))

    st.divider()

    # ── Beeswarm plots per class ──────────────────────────────────────────────
    st.markdown("### 4️⃣  SHAP Beeswarm — Deep Dive per Fault Class")
    st.caption(
        "Each dot is one record. Colour = feature value (high = red, low = blue). "
        "X-axis = SHAP value: positive pushes model toward this class, "
        "negative pushes away. Shows both *which* features matter and *how* "
        "they influence predictions."
    )
    class_options = list(LABEL_MAP.values())
    selected_class = st.selectbox(
        "Select fault class for beeswarm:",
        options=class_options,
        index=1,
        key="beeswarm_class",
    )
    class_idx = {v: k for k, v in LABEL_MAP.items()}[selected_class]

    try:
        fig = plot_shap_beeswarm(shap_values, X_sample, class_idx)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Beeswarm error: {e}")

    st.divider()

    # ── Static model feature importances (cross-check) ────────────────────────
    st.markdown("### 5️⃣  Model-Level Feature Importances (Cross-Check)")
    st.caption(
        "Random Forest's built-in Gini importance — class-agnostic, for cross-validation "
        "against SHAP. Consistent top features in both methods increase confidence "
        "in the explanations (robustness check for thesis §6)."
    )
    try:
        rf_imp = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False).head(10)
        st.bar_chart(rf_imp.rename_axis("Feature").reset_index(name="Gini Importance").set_index("Feature"))
    except Exception as e:
        st.error(f"Feature importance error: {e}")

    st.divider()
    st.caption(
        f"**{THESIS_META['title']}**  ·  {THESIS_META['xai']}  ·  "
        f"Framework: {THESIS_META['framework']}"
    )
