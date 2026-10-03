"""
pages/02_fault_analysis.py
--------------------------
Page 2 — Fault Detection & Classification Analysis

Thesis alignment
----------------
Contribution 1 (Multi-fault classification framework):
This page provides the evidence that the classifier can distinguish
between Healthy, Bearing Wear, Overheating/Fouling, and Power
Inefficiency. The confusion matrix, per-class metrics, and timeline
directly demonstrate classification performance — the core technical
contribution of the thesis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import LABEL_MAP, THESIS_META
from model_utils import compute_per_class_accuracy
from visualizations import (
    plot_confusion_matrix,
    plot_fault_distribution,
    plot_fault_timeline,
    style_metrics_table,
)


def render(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render the Fault Detection & Classification Analysis page."""

    st.markdown("## 🔍 Fault Detection & Classification Analysis")
    st.markdown(
        "_Performance evaluation of the multi-fault Random Forest classifier. "
        "This page provides evidence for **Thesis Contribution 1**: "
        "a working multi-fault classification framework for HVAC chillers._"
    )
    st.divider()

    if filtered_df.empty:
        st.warning("No records match the current filters. Adjust the sidebar.")
        return

    # ── Per-class performance metrics ─────────────────────────────────────────
    st.markdown("### 📐 Classification Performance Metrics")
    st.caption(
        "Precision, Recall, and F1 Score per fault class. "
        "Colour gradient: 🟢 high → 🔴 low. "
        "These values directly populate Table X of the thesis results chapter."
    )
    try:
        metrics_df = compute_per_class_accuracy(filtered_df)
        styled = style_metrics_table(metrics_df)
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # Interpretation note (academic value)
        best_f1  = metrics_df.loc[metrics_df["F1 Score"].idxmax()]
        worst_f1 = metrics_df.loc[metrics_df["F1 Score"].idxmin()]
        st.info(
            f"**Best classified:** {best_f1['Class']} (F1 = {best_f1['F1 Score']:.3f})  |  "
            f"**Most challenging:** {worst_f1['Class']} (F1 = {worst_f1['F1 Score']:.3f}).  "
            "Lower F1 for minority fault classes is consistent with class imbalance "
            "effects discussed in §5 of the thesis proposal."
        )
    except Exception as e:
        st.error(f"Could not compute metrics: {e}")

    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🧩 Confusion Matrix")
    st.caption(
        "Each cell shows the raw count and the row-normalised percentage. "
        "Diagonal = correct predictions; off-diagonal = misclassifications. "
        "Overlapping fault signatures cause cross-class confusion "
        "(discussed in §5: Findings)."
    )
    try:
        fig = plot_confusion_matrix(filtered_df)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Confusion matrix error: {e}")

    st.divider()

    # ── Fault distribution ────────────────────────────────────────────────────
    st.markdown("### 📊 Predicted Fault Distribution")
    st.caption(
        "Class imbalance is a known challenge for this dataset (see §4: Research Design). "
        "A dominant Healthy class is expected in real HVAC operation."
    )
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = plot_fault_distribution(filtered_df)
        st.pyplot(fig)
        plt.close(fig)
    with col_table:
        dist_df = (
            filtered_df["Predicted_Class_Name"]
            .value_counts()
            .rename_axis("Fault Class")
            .reset_index(name="Count")
        )
        dist_df["Share (%)"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(1)
        st.dataframe(dist_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Fault timeline ────────────────────────────────────────────────────────
    st.markdown("### 🕐 Classification Timeline (Last 200 Records)")
    st.caption(
        "Temporal distribution of fault predictions. "
        "Clustered fault events may indicate a degradation episode; "
        "intermittent single-point events may be sensor noise."
    )
    try:
        fig = plot_fault_timeline(filtered_df, tail_n=200)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Timeline error: {e}")

    st.divider()

    # ── Raw prediction table ──────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Prediction Log")
    with st.expander("Show full prediction table", expanded=False):
        display_cols = (
            ["Timestamp", "Actual_Class_Name", "Predicted_Class_Name"]
            + [c for c in filtered_df.columns if c.startswith("Prob_")]
        )
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[display_cols].sort_values("Timestamp", ascending=False),
            use_container_width=True,
        )

    st.divider()
    st.caption(
        f"**{THESIS_META['title']}**  ·  {THESIS_META['model']}  ·  "
        f"Framework: {THESIS_META['framework']}"
    )
