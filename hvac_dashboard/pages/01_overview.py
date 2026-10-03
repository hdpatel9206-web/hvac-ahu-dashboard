"""
pages/01_overview.py
--------------------
Page 1 — System Overview

Thesis alignment
----------------
Contribution 4 (Practical usability): This is the landing page for
facility managers. At a glance they see the current fault status,
headline KPIs, sensor readings, and a fault timeline — without
needing any ML expertise. Directly satisfies Research Question
section 3: "supports maintenance decision-making through visualization."
"""

import pandas as pd
import streamlit as st

from config import FAULT_STYLE, RECOMMENDATIONS, SENSOR_META, THESIS_META
from data_loader import compute_summary_stats
from visualizations import plot_fault_distribution, plot_fault_timeline, plot_sensor_trends


def render(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render the System Overview page."""

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown("## 🏢 System Overview")
    st.markdown(
        "_Real-time operational status and historical fault summary "
        "for the monitored HVAC chiller system._"
    )
    st.divider()

    # ── Current system status banner ─────────────────────────────────────────
    source_df = filtered_df if not filtered_df.empty else df
    latest    = source_df.iloc[-1]
    current_fault = latest["Predicted_Class_Name"]

    status_fn = getattr(st, FAULT_STYLE.get(current_fault, ("info", ""))[0])
    status_fn(f"### Current Fault Status: **{current_fault}**")

    # Recommendation box
    rec = RECOMMENDATIONS.get(current_fault, "Review system manually.")
    st.info(f"**Maintenance Recommendation:**\n\n{rec}")

    st.divider()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Summary KPIs  _(filtered period)_")
    stats = compute_summary_stats(filtered_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records",       f"{stats['total_records']:,}")
    col2.metric("Fault Rate",          f"{stats['fault_rate_pct']} %")
    col3.metric("Healthy Records",     f"{stats['healthy_pct']} %")
    col4.metric("Dominant Fault",      stats['most_common_fault'])
    col5.metric("Alert Events",        f"{stats['alert_count']:,}")

    st.divider()

    # ── Live sensor readings ──────────────────────────────────────────────────
    st.markdown("#### 🌡️ Current Sensor Readings")
    sensor_cols = [c for c in SENSOR_META if c in df.columns]
    if sensor_cols:
        metric_cols = st.columns(len(sensor_cols))
        for col_st, sensor in zip(metric_cols, sensor_cols):
            label, unit = SENSOR_META[sensor]
            val     = latest[sensor]
            # Compute delta vs rolling mean of last 20 records
            hist_mean = source_df[sensor].tail(20).mean()
            delta = val - hist_mean
            col_st.metric(
                label=f"{label} ({unit})",
                value=f"{val:.2f}",
                delta=f"{delta:+.2f} vs 20-rec avg",
            )
    else:
        st.info("No recognised sensor columns found.")

    st.divider()

    # ── Fault distribution chart ──────────────────────────────────────────────
    st.markdown("#### 📉 Fault Distribution (Filtered Period)")
    if filtered_df.empty:
        st.warning("No records match the current filters.")
    else:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = plot_fault_distribution(filtered_df)
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        with col_b:
            count_df = (
                filtered_df["Predicted_Class_Name"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Fault Class", "Predicted_Class_Name": "Count"})
            )
            st.dataframe(count_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Fault timeline ────────────────────────────────────────────────────────
    st.markdown("#### 🕐 Fault Classification Timeline")
    st.caption(
        "Each dot represents one record. The Y-axis shows the predicted fault class. "
        "Red-shaded periods indicate non-healthy predictions."
    )
    if not filtered_df.empty:
        fig = plot_fault_timeline(filtered_df)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        st.warning("No records to display.")

    st.divider()

    # ── Sensor trends ─────────────────────────────────────────────────────────
    st.markdown("#### 📈 Sensor Reading Trends")
    st.caption(
        "Shaded red regions indicate timesteps where a fault was predicted. "
        "Useful for correlating sensor spikes with fault events."
    )
    sensor_cols = [c for c in SENSOR_META if c in df.columns]
    if not filtered_df.empty and sensor_cols:
        n_tail = min(len(filtered_df), 300)
        fig = plot_sensor_trends(filtered_df.tail(n_tail), sensor_cols)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        st.info("No sensor data available for the selected filters.")

    # ── Footer note ───────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"**{THESIS_META['title']}**  ·  {THESIS_META['model']}  ·  "
        f"Framework: {THESIS_META['framework']}"
    )
