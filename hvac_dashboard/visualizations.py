"""
visualizations.py
-----------------
All Matplotlib / Seaborn / SHAP chart rendering functions.

Every function returns a ``matplotlib.figure.Figure`` so the caller
can pass it to ``st.pyplot(fig)`` and then ``plt.close(fig)``.
No Streamlit calls inside this module — pure visualisation logic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import FAULT_COLOURS, FAULT_STYLE, LABEL_MAP, SHAP_TOP_N

# ── Shared style ─────────────────────────────────────────────────────────────
PALETTE = list(FAULT_STYLE[v][1] for v in LABEL_MAP.values())

def _apply_thesis_style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply a clean, consistent academic style to a single Axes."""
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color="#DEE2E6", linewidth=0.7, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CED4DA")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#212529")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#495057")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#495057")
    ax.tick_params(colors="#495057", labelsize=9)


# ── Fault distribution ───────────────────────────────────────────────────────

def plot_fault_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of predicted fault class counts.
    Annotates each bar with count and percentage.
    """
    counts = df["Predicted_Class_Name"].value_counts().reindex(
        list(LABEL_MAP.values()), fill_value=0
    )
    total = counts.sum() or 1

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(
        counts.index,
        counts.values,
        color=[FAULT_STYLE[k][1] for k in counts.index],
        edgecolor="white",
        height=0.55,
    )
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center",
            fontsize=9,
            color="#495057",
        )
    _apply_thesis_style(ax, title="Predicted Fault Distribution", xlabel="Record Count")
    ax.set_xlim(0, counts.max() * 1.25)
    fig.tight_layout()
    return fig


# ── Fault timeline ───────────────────────────────────────────────────────────

def plot_fault_timeline(df: pd.DataFrame, tail_n: int = 200) -> plt.Figure:
    """
    Scatter plot of predicted fault class over time.
    Each class is a distinct coloured row, making pattern shifts visible.
    """
    plot_df = df[["Timestamp", "Predicted_Class_Name"]].tail(tail_n).copy()
    class_order = list(LABEL_MAP.values())
    plot_df["y"] = plot_df["Predicted_Class_Name"].map(
        {v: i for i, v in enumerate(class_order)}
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    for cls in class_order:
        sub = plot_df[plot_df["Predicted_Class_Name"] == cls]
        ax.scatter(
            sub["Timestamp"],
            sub["y"],
            c=FAULT_STYLE[cls][1],
            s=18,
            alpha=0.75,
            label=cls,
            zorder=3,
        )

    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(class_order, fontsize=9)
    ax.set_xlabel("Time", fontsize=10, color="#495057")
    _apply_thesis_style(ax, title=f"Fault Classification Timeline (last {tail_n} records)")
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", visible=False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


# ── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Annotated confusion matrix heatmap with per-cell accuracy %.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    labels     = list(LABEL_MAP.keys())
    label_names = list(LABEL_MAP.values())

    cm = sk_cm(df["Actual_Label"], df["Predicted_Label"], labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    annot = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)" for j in range(len(labels))]
        for i in range(len(labels))
    ])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#DEE2E6",
        cbar_kws={"label": "Row %"},
        vmin=0,
        vmax=100,
    )
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
    ax.set_title("Multi-Fault Classification — Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return fig


# ── Per-class metrics table ───────────────────────────────────────────────────

def style_metrics_table(metrics_df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Return a Pandas Styler that colour-grades F1 Score cells
    (green = high, red = low) for quick visual assessment.
    """
    return (
        metrics_df.style
        .background_gradient(subset=["Precision", "Recall", "F1 Score"],
                             cmap="RdYlGn", vmin=0, vmax=1)
        .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"})
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#1A3A5C"),
                                          ("color", "white"),
                                          ("font-weight", "bold"),
                                          ("text-align", "center")]},
        ])
    )


# ── Sensor trends ─────────────────────────────────────────────────────────────

def plot_sensor_trends(df: pd.DataFrame, sensor_cols: list[str]) -> plt.Figure:
    """
    Multi-panel line chart of sensor readings over time.
    Each sensor gets its own subplot with a shaded fault region overlay.
    """
    n = len(sensor_cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colours = ["#2980b9", "#e74c3c", "#27ae60"]

    for ax, col, colour in zip(axes, sensor_cols, colours):
        plot_df = df[["Timestamp", col, "Predicted_Class_Name"]].copy()

        # Shade fault regions
        for _, row in plot_df.iterrows():
            if row["Predicted_Class_Name"] != "Healthy":
                ax.axvspan(
                    row["Timestamp"] - pd.Timedelta(minutes=1),
                    row["Timestamp"] + pd.Timedelta(minutes=1),
                    alpha=0.08,
                    color="#e74c3c",
                    linewidth=0,
                )

        ax.plot(
            plot_df["Timestamp"],
            plot_df[col],
            color=colour,
            linewidth=1.2,
            alpha=0.85,
        )
        label, unit = col.replace("_", " "), col.split("_")[-1]
        _apply_thesis_style(ax, ylabel=f"{label} ({unit})")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[-1].set_xlabel("Timestamp", fontsize=10, color="#495057")
    fig.suptitle("Sensor Readings Over Time  (🔴 shaded = fault event)", fontsize=11, fontweight="bold", y=1.01)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


# ── SHAP: overall summary bar ─────────────────────────────────────────────────

def plot_shap_summary_bar(shap_values, X_sample: pd.DataFrame, top_n: int = SHAP_TOP_N) -> plt.Figure:
    """
    Multi-class SHAP summary bar chart — mean |SHAP| per feature per class.
    Grouped bars, one group per feature, one bar per class.
    """
    import shap as shap_lib

    fig = plt.figure(figsize=(10, 5))
    shap_lib.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        class_names=list(LABEL_MAP.values()),
        max_display=top_n,
        show=False,
        color=PALETTE,
    )
    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Importance — Mean |SHAP| per Class",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    fig.tight_layout()
    return fig


# ── SHAP: beeswarm for one class ──────────────────────────────────────────────

def plot_shap_beeswarm(shap_values, X_sample: pd.DataFrame, class_idx: int) -> plt.Figure:
    """
    SHAP beeswarm plot for a single fault class.
    Shows *how* each feature pushes the model toward or away from that class.
    """
    import shap as shap_lib

    fig = plt.figure(figsize=(9, 5))
    # shap_values is a list; index by class
    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    shap_lib.summary_plot(sv, X_sample, show=False, max_display=SHAP_TOP_N)
    ax = plt.gca()
    class_name = LABEL_MAP.get(class_idx, f"Class {class_idx}")
    ax.set_title(
        f"SHAP Beeswarm — Class: {class_name}",
        fontsize=12, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    return fig


# ── SHAP: per-class mean |SHAP| heatmap ──────────────────────────────────────

def plot_shap_heatmap(mean_abs_df: pd.DataFrame, top_n: int = SHAP_TOP_N) -> plt.Figure:
    """
    Heatmap of mean |SHAP| for top N features × all fault classes.
    Thesis-relevant: shows which sensors drive which faults — directly
    supporting the research hypothesis (vibration → bearing, temp → thermal).
    """
    top_features = mean_abs_df.mean(axis=1).nlargest(top_n).index
    plot_df = mean_abs_df.loc[top_features]

    fig, ax = plt.subplots(figsize=(8, 0.45 * top_n + 1.5))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.4,
        linecolor="#DEE2E6",
        cbar_kws={"label": "Mean |SHAP|", "shrink": 0.8},
    )
    ax.set_title(
        "Feature–Fault Attribution Heatmap  (Mean |SHAP| Value)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Fault Class",  fontsize=10)
    ax.set_ylabel("Feature",      fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return fig


# ── SHAP: single-prediction force-like waterfall ─────────────────────────────

def plot_shap_waterfall_single(
    shap_values,
    X_single: pd.DataFrame,
    predicted_class_idx: int,
    top_n: int = 8,
) -> plt.Figure:
    """
    Waterfall-style bar chart of SHAP contributions for a single prediction.
    Positive bars push toward the predicted class; negative bars push away.
    Directly supports Contribution 2 (interpretable predictions).
    """
    if isinstance(shap_values, list):
        sv = shap_values[predicted_class_idx][0]
    else:
        sv = shap_values[0]

    features = X_single.columns.tolist()
    vals     = list(zip(features, sv))
    vals.sort(key=lambda x: abs(x[1]), reverse=True)
    vals = vals[:top_n]

    names = [v[0] for v in vals]
    contribs = [v[1] for v in vals]
    colours  = ["#27ae60" if c > 0 else "#e74c3c" for c in contribs]

    fig, ax = plt.subplots(figsize=(8, 0.5 * top_n + 1.5))
    bars = ax.barh(names[::-1], contribs[::-1], color=colours[::-1], edgecolor="white", height=0.55)

    for bar, val in zip(bars, contribs[::-1]):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            color="#212529",
        )

    ax.axvline(0, color="#495057", linewidth=0.8)
    class_name = LABEL_MAP.get(predicted_class_idx, "Unknown")
    _apply_thesis_style(
        ax,
        title=f"SHAP Explanation — Predicted: {class_name}",
        xlabel="SHAP Contribution (positive = pushes toward this class)",
    )
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ── Live prediction: probability gauge bars ──────────────────────────────────

def plot_prediction_probabilities(proba_dict: dict[str, float]) -> plt.Figure:
    """
    Horizontal bar chart of prediction probabilities for all classes.
    The predicted class bar is highlighted.
    """
    names  = list(proba_dict.keys())
    values = [proba_dict[n] for n in names]
    max_v  = max(values)
    colours = [
        FAULT_STYLE.get(n, ("info", "#2980b9"))[1]
        if proba_dict[n] == max_v
        else "#B0BEC5"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(names, values, color=colours, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center", fontsize=9.5, color="#212529",
        )
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _apply_thesis_style(ax, title="Fault Class Probabilities", xlabel="Probability")
    fig.tight_layout()
    return fig


# ── Input vs historical distribution ─────────────────────────────────────────

def plot_input_vs_historical(
    df: pd.DataFrame,
    feature_cols: list[str],
    input_values: dict[str, float],
    max_features: int = 6,
) -> plt.Figure:
    """
    Overlapping histogram: historical distribution (blue) + live input marker (red).
    Shows facility managers where the current reading sits in the historical range.
    """
    display = feature_cols[:max_features]
    ncols   = min(3, len(display))
    nrows   = (len(display) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for i, feat in enumerate(display):
        ax = axes_flat[i]
        ax.hist(df[feat].dropna(), bins=30, alpha=0.65, color="#2980b9", label="Historical")
        ax.axvline(input_values[feat], color="#e74c3c", linewidth=2, linestyle="--", label="Live input")
        _apply_thesis_style(ax, title=feat.replace("_", " "))
        if i == 0:
            ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(len(display), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Live Sensor Input vs Historical Distribution", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
