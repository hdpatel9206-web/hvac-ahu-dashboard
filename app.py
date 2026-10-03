"""
AHU Multi-Fault Detection Dashboard — Final Corrected Version
==============================================================
Thesis: Multi-Fault Detection and Diagnosis for Smart Building AHUs:
        An Explainable Machine Learning Approach with Cross-Building Validation

Student  : Harshil Patel
Programme: M.Sc. Smart Building Technologies (M_SSBT_W24)
bbw Hochschule Berlin — University of Applied Sciences
Supervisors: Prof. Farshi Hossein · Prof. Dr. Juan Ocampo

PRIMARY MODEL (run_20260409_022716_ASHRAE_LBNL):
    In-sample macro-F1 : 0.9923  CI [0.9913, 0.9933]
    Cross-building F1  : 0.331   Gap = 66.13 pp
    Binary F1 CB       : 0.415
    Fan speed PSI      : 2.6321  OA temperature PSI : 2.4517
    Binary F1 (10% adapt): 0.992
    Real-world PSI ratio : 17.2x
    Source: results/run_20260409_022716_ASHRAE_LBNL/results_summary.json

Self-Healing Loop (extension):
    CBF violations (real BMS)  : 0 / 203,040 timesteps (5 datasets)
    CBF violations (simulation): 0 / 43,200 minutes (30-day closed-loop)
    Days 24-26 GREEN gate      : binary F1 = 0.610, 4-class F1 = 0.433
    Verification               : 4/4 PASS in 1.37s, seed=42, 0/600 adversarial
    Note: simulation uses calibrated surrogate (MockFDD), not rf_model.pkl

Five pages:
    Page 1 — System Overview          (PSI gate + KPIs + fault timeline)
    Page 2 — Fault Analysis           (confusion matrix + per-class metrics)
    Page 3 — Explainability (SHAP)    (pre-computed SHAP ranking + PSI vs SHAP table)
    Page 4 — Cost Calculator          (probability-weighted energy cost model)
    Page 5 — Self-Healing Loop        (CBF results + PSI tracking + RL metrics)

Live single-sample prediction is not included: the canonical 4-class model
(119 MB) exceeds GitHub's file limit. No model file is loaded.

Run:  streamlit run app.py
Deps: streamlit pandas numpy matplotlib seaborn
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH           = Path("data/chiller_predictions.csv")
SHAP_RESULTS_PATH   = Path("results/shap_values.json")

# SHAP sample cap — only used if user requests live computation
TIMELINE_TAIL   = 200

LABEL_MAP: dict[int, str] = {
    0: "Healthy",
    1: "Sensor Bias",
    2: "Valve Fault",
    3: "Damper Fault",
}

FAULT_CONFIG: dict[str, tuple[str, str, str]] = {
    "Healthy":      ("success", "#27ae60", "No action required"),
    "Sensor Bias":  ("warning", "#f39c12", "Schedule sensor inspection within 7 days"),
    "Valve Fault":  ("error",   "#e74c3c", "Immediate valve inspection required"),
    "Damper Fault": ("warning", "#e67e22", "Inspect damper actuator within 3 days"),
}

PROB_COLS: dict[int, str] = {
    0: "Prob_Healthy",
    1: "Prob_Sensor_Bias",
    2: "Prob_Valve_Fault",
    3: "Prob_Damper_Fault",
}

# ── Primary model anchor metrics ───────────────────────────────────────────
# Source: results/run_20260409_022716_ASHRAE_LBNL/results_summary.json
THESIS_METRICS = dict(
    f1_insample    = 0.9923,
    ci_lo          = 0.9913,
    ci_hi          = 0.9933,
    f1_crossbldg   = 0.331,
    gap_pp         = 66.13,
    binary_cb      = 0.415,   # FIX-5: was 0.628 (that was the rejected variant)
    fan_psi        = 2.6321,  # exact from results_summary.json
    oa_psi         = 2.4517,  # exact from results_summary.json
    binary_adapt   = 0.992,
    rw_ratio       = 17.2,
    cbf_violations = 0,
    real_timesteps = 203_040, # 5 real BMS datasets
    sim_minutes    = 43_200,  # 30-day closed-loop simulation
    green_binary   = 0.610,
    green_4class   = 0.433,
)

# ── PSI sensor values — from results_summary.json psi_top_features ─────────
PSI_VALUES = {
    "SA_Temp":    0.8401,
    "OA_Temp":    2.4517,
    "MA_Temp":    0.5122,
    "RA_Temp":    0.7136,
    "Fan_Status": 0.0100,
    "Fan_Speed":  2.6321,
    "OA_Damper":  0.4811,
    "RA_Damper":  1.4386,
    "CC_Valve":   0.88,    # supplementary psi computation
    "HC_Valve":   0.49,    # supplementary psi computation
    "Occupancy":  0.0002,  # supplementary psi computation
}

# GREEN < 0.20 · AMBER 0.20–0.50 · RED > 0.50 (thesis framework figure;
# Yurdakul & Naranjo, 2019). Every PSI classification goes through psi_gate().
PSI_THRESHOLDS = {"GREEN": 0.20, "AMBER": 0.20, "RED": 0.50}
PSI_GATE_COLOURS = {"RED": "#e74c3c", "AMBER": "#f39c12", "GREEN": "#27ae60"}
PSI_BANDS_TEXT = (f"GREEN < {PSI_THRESHOLDS['GREEN']:.2f} · "
                  f"AMBER {PSI_THRESHOLDS['AMBER']:.2f}–{PSI_THRESHOLDS['RED']:.2f} · "
                  f"RED > {PSI_THRESHOLDS['RED']:.2f}")


def psi_gate(psi: float) -> str:
    """GREEN if PSI < GREEN, RED if PSI > RED (strict), otherwise AMBER."""
    if psi > PSI_THRESHOLDS["RED"]:
        return "RED"
    if psi < PSI_THRESHOLDS["GREEN"]:
        return "GREEN"
    return "AMBER"

# ── Maintenance recommendations ────────────────────────────────────────────
# FIX-7: removed incorrect "SA_Temp_std_30min = 0.0378" claim
# Top feature by impurity importance: Return Air Damper (0.0932), Table 4.4
RECOMMENDATIONS = {
    "Healthy": (
        "System operating within normal parameters. No maintenance action required. "
        "Continue scheduled inspection every 30 days."
    ),
    "Sensor Bias": (
        "Outdoor air temperature sensor bias detected. Inspect OA sensor calibration "
        "and verify readings against reference instrument. Schedule within 7 days. "
        "SHAP attribution: OA temperature features are the primary drivers (Table 4.4)."
    ),
    "Valve Fault": (
        "Heating or cooling coil valve fault detected. Inspect valve actuator and seals. "
        "Risk of simultaneous heating and cooling — energy waste 15%. Immediate inspection. "
        "Top attribution driver: Return Air Damper Control Signal (importance 0.0932)."
    ),
    "Damper Fault": (
        "Damper actuator fault detected. Inspect return air and outdoor air damper "
        "positions and actuator linkage. Investigate within 3 days. "
        "RA_Damper PSI = 1.44 (RED) — building-specific commissioning difference identified."
    ),
}

# ── Cost model parameters ──────────────────────────────────────────────────
COST_PARAMS: dict[str, dict] = {
    "Healthy": dict(
        energy_loss_pct=0, downtime_risk_days=0,
        repair_now_parts=0, repair_now_labour_h=0,
        repair_reactive_parts=0, repair_reactive_labour_h=0,
        escalation_weeks=0,
        source="No corrective action required.",
        interpretation="System healthy. No cost impact.",
    ),
    "Sensor Bias": dict(
        energy_loss_pct=3, downtime_risk_days=0.5,
        repair_now_parts=900, repair_now_labour_h=4,
        repair_reactive_parts=18000, repair_reactive_labour_h=16,
        escalation_weeks=3,
        source="Sensor recalibration EUR 800-1,200 scheduled vs EUR 15,000+ emergency. DOE (2017).",
        interpretation=(
            "OA sensor bias causes incorrect ventilation control. "
            "SHAP attribution confirms OA temperature features as primary drivers. "
            "PSI for OA temperature = 2.45 (RED gate) — distribution shift confirmed."
        ),
    ),
    "Valve Fault": dict(
        energy_loss_pct=15, downtime_risk_days=1.5,
        repair_now_parts=800, repair_now_labour_h=6,
        repair_reactive_parts=7500, repair_reactive_labour_h=24,
        escalation_weeks=1,
        source="Energy penalty 15%: simultaneous heating and cooling. Zhao et al. (2019).",
        interpretation=(
            "Valve leakage causes simultaneous heating and cooling. "
            "Top SHAP feature: Return Air Damper Control Signal (importance 0.0932, Table 4.4). "
            "Thermodynamic fault — simulation augmentation improves cross-building F1 by +1.9pp."
        ),
    ),
    "Damper Fault": dict(
        energy_loss_pct=10, downtime_risk_days=0.2,
        repair_now_parts=500, repair_now_labour_h=3,
        repair_reactive_parts=4000, repair_reactive_labour_h=10,
        escalation_weeks=6,
        source="Damper actuator replacement EUR 400-700 scheduled. DOE (2017).",
        interpretation=(
            "Damper faults cause incorrect airflow ratios. RA_Damper PSI = 1.44 (RED) "
            "indicates the return air damper distribution has shifted significantly "
            "between training and deployment buildings."
        ),
    ),
}

# ── Verified experiment data (from JSON files) ─────────────────────────────

# FIX-4: real values from learning_curve_results.json (non-monotonic, as expected)
LEARNING_CURVE_PCTS  = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LEARNING_CURVE_F1_CB = [0.5463, 0.5473, 0.5155, 0.5348, 0.5472,
                         0.5317, 0.5309, 0.5306, 0.4983, 0.5224, 0.5592]

# FIX-3: real values from finetune_fixed_results.json
# 4-class macro-F1 converges on 0.75 ceiling (Class 1 absent from test set)
FINETUNE_PCTS      = [0,      5,      10,     15,     20,     30,     50    ]
FINETUNE_F1_4CLASS = [0.3116, 0.7395, 0.7440, 0.7468, 0.7474, 0.7485, 0.7498]
FINETUNE_F1_BINARY = [0.4154, 0.9860, 0.9920, 0.9958, 0.9965, 0.9980, 0.9997]
FINETUNE_THEO_MAX  = 0.75  # (1+0+1+1)/4: Class 1 absent from withheld buildings

# ═══════════════════════════════════════════════════════════════════════════
# 2. CACHED LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading prediction data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Timestamp" not in df.columns:
        df.insert(0, "Timestamp", pd.date_range(
            start="2024-01-01", periods=len(df), freq="1min"))
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "Predicted_Class" not in df.columns:
        df["Predicted_Class"] = df["Predicted_Label"].map(LABEL_MAP)
    if "Actual_Class" not in df.columns:
        df["Actual_Class"] = df.get(
            "Actual_Label", pd.Series(0, index=df.index)).map(LABEL_MAP)
    for col in PROB_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df.sort_values("Timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def filter_df(df, start, end, faults):
    mask = (
        (df["Timestamp"].dt.date >= start)
        & (df["Timestamp"].dt.date <= end)
        & (df["Predicted_Class"].isin(faults))
    )
    return df[mask].reset_index(drop=True)


def kpis(df):
    if df.empty:
        return dict(total=0, fault_rate=0.0, healthy_pct=0.0, top_fault="N/A")
    n       = len(df)
    healthy = (df["Predicted_Class"] == "Healthy").sum()
    faults  = df[df["Predicted_Class"] != "Healthy"]["Predicted_Class"]
    top     = faults.value_counts().idxmax() if len(faults) > 0 else "None"
    return dict(
        total       = n,
        fault_rate  = round((n - healthy) / n * 100, 1),
        healthy_pct = round(healthy / n * 100, 1),
        top_fault   = top,
    )


def per_class_metrics(df):
    from sklearn.metrics import classification_report
    rpt = classification_report(
        df["Actual_Label"], df["Predicted_Label"],
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls in LABEL_MAP.values():
        r = rpt.get(cls, {})
        rows.append({
            "Fault Class": cls,
            "Precision":   round(r.get("precision", 0), 4),
            "Recall":      round(r.get("recall",    0), 4),
            "F1 Score":    round(r.get("f1-score",  0), 4),
            "Support":     int(r.get("support",     0)),
        })
    return pd.DataFrame(rows)


def compute_cost(fault_class, fault_probability, capacity_kw,
                 electricity_eur_kwh, hours_per_day, labour_rate=85.0):
    p      = fault_probability
    params = COST_PARAMS.get(fault_class, COST_PARAMS["Healthy"])
    monthly_kwh         = capacity_kw * hours_per_day * 30
    base_energy_monthly = monthly_kwh * electricity_eur_kwh
    energy_penalty      = base_energy_monthly * (params["energy_loss_pct"] / 100) * p
    downtime_cost       = params["downtime_risk_days"] * hours_per_day * 150 * p
    ep_amort            = params["repair_reactive_parts"] * p / 12
    el_amort            = params["repair_reactive_labour_h"] * labour_rate * p / 12
    total_do_nothing    = energy_penalty + downtime_cost + ep_amort + el_amort
    repair_now_total    = (params["repair_now_parts"] +
                           params["repair_now_labour_h"] * labour_rate) * p
    repair_now_monthly  = repair_now_total / 12
    net_saving          = max(0.0, total_do_nothing - repair_now_monthly)
    payback             = repair_now_total / (net_saving * 12) if net_saving > 0 else None
    return dict(
        base_energy_monthly   = round(base_energy_monthly, 2),
        energy_penalty        = round(energy_penalty, 2),
        downtime_cost         = round(downtime_cost, 2),
        emergency_parts_amort = round(ep_amort, 2),
        emergency_labour      = round(el_amort, 2),
        total_do_nothing      = round(total_do_nothing, 2),
        repair_now_total      = round(repair_now_total, 2),
        repair_now_monthly    = round(repair_now_monthly, 2),
        net_monthly_saving    = round(net_saving, 2),
        payback_months        = round(payback, 1) if payback else None,
        params                = params,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

MPL_STYLE = {
    "axes.facecolor": "#F8F9FA", "figure.facecolor": "white",
    "grid.color": "#DEE2E6",     "grid.linewidth": 0.6,
    "axes.spines.top": False,    "axes.spines.right": False,
    "axes.spines.left": False,   "axes.edgecolor": "#CED4DA",
    "text.color": "#212529",     "axes.labelcolor": "#495057",
    "xtick.color": "#495057",    "ytick.color": "#495057",
}


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(MPL_STYLE["axes.facecolor"])
    ax.grid(axis="y", linewidth=0.6, color="#DEE2E6", linestyle="--")
    ax.grid(axis="x", visible=False)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    if title:  ax.set_title(title,  fontsize=11.5, fontweight="bold", pad=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(labelsize=8.5)


def _show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def fig_fault_bar(df):
    counts = df["Predicted_Class"].value_counts().reindex(
        LABEL_MAP.values(), fill_value=0)
    total  = counts.sum() or 1
    clrs   = [FAULT_CONFIG[k][1] for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(counts.index, counts.values,
                   color=clrs, edgecolor="white", height=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + total * 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({val/total*100:.1f}%)",
                va="center", fontsize=8.5)
    _style(ax, title="Predicted Fault Distribution", xlabel="Record Count")
    ax.set_xlim(0, counts.max() * 1.3)
    fig.tight_layout()
    return fig


def fig_timeline(df):
    order = list(LABEL_MAP.values())
    y_map = {v: i for i, v in enumerate(order)}
    plot  = df[["Timestamp", "Predicted_Class"]].tail(TIMELINE_TAIL).copy()
    plot["y"] = plot["Predicted_Class"].map(y_map)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for cls in order:
        sub = plot[plot["Predicted_Class"] == cls]
        ax.scatter(sub["Timestamp"], sub["y"],
                   c=FAULT_CONFIG[cls][1], s=16, alpha=0.75, label=cls, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.5)
    _style(ax, title=f"Fault Timeline (last {TIMELINE_TAIL} records)")
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.5, linestyle="--")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def fig_psi_bar():
    """PSI bar chart — values from results_summary.json psi_top_features."""
    sensors = list(PSI_VALUES.keys())
    values  = list(PSI_VALUES.values())
    clrs    = [PSI_GATE_COLOURS[psi_gate(v)] for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(sensors, values, color=clrs, edgecolor="white", height=0.55)
    ax.axvline(PSI_THRESHOLDS["AMBER"], color="#f39c12", lw=1.5, ls="--",
               label=f"AMBER (≥{PSI_THRESHOLDS['AMBER']:.2f})")
    ax.axvline(PSI_THRESHOLDS["RED"],   color="#e74c3c", lw=1.5, ls="--",
               label=f"RED (>{PSI_THRESHOLDS['RED']:.2f})")
    for bar, val in zip(bars, values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)
    ax.legend(fontsize=8.5, loc="lower right")
    _style(ax,
           title="Population Stability Index — All 11 Sensors (Primary Model)",
           xlabel="PSI Value")
    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()
    return fig


def fig_confusion_matrix(df):
    from sklearn.metrics import confusion_matrix as sk_cm
    labels = list(LABEL_MAP.keys())
    names  = list(LABEL_MAP.values())
    cm     = sk_cm(df["Actual_Label"], df["Predicted_Label"], labels=labels)
    row_n  = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = cm.astype(float) / row_n * 100
    annot  = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)"
                        for j in range(4)] for i in range(4)])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.4, linecolor="#DEE2E6",
                cbar_kws={"label": "Row %"}, vmin=0, vmax=100)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=10, labelpad=8)
    ax.set_title("Multi-Fault Confusion Matrix", fontsize=11.5,
                 fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=18, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)
    fig.tight_layout()
    return fig


# Neither chart below is simulation output: self_healing_loop.py saves no PSI
# trajectory, so the curve is a formula plus seeded random noise.
PSI_SCHEMATIC_TITLE = "Illustrative PSI trajectory (schematic, not simulation output)"


def fig_psi_simulation():
    """Schematic 30-day PSI trajectory (formula + noise, not simulation output)."""
    np.random.seed(42)
    days    = np.linspace(0, 30, 43200)
    fan_psi = np.where(
        days < 5,
        np.random.uniform(0, 0.05, 43200),
        np.where(
            days < 15,
            (days - 5) / 10 * 15 + np.random.normal(0, 0.3, 43200),
            np.maximum(0, 14 - (days - 15) * 0.5 +
                       np.random.normal(0, 0.8, 43200))
        )
    )
    fan_smooth = pd.Series(fan_psi).ewm(alpha=0.001).mean().values

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(days, fan_smooth, color="#e74c3c", lw=1.2,
            label="Fan Speed PSI (smoothed)")
    ax.axhline(PSI_THRESHOLDS["AMBER"], color="#f39c12", lw=1.4, ls="--",
               label=f"AMBER gate ({PSI_THRESHOLDS['AMBER']:.2f})")
    ax.axhline(PSI_THRESHOLDS["RED"], color="#e74c3c", lw=1.4, ls="--",
               label=f"RED gate ({PSI_THRESHOLDS['RED']:.2f})")
    ax.axvspan(0,  5,  alpha=0.08, color="#27ae60", label="Phase 1: Clean")
    ax.axvspan(5,  15, alpha=0.08, color="#e74c3c", label="Phase 2: Shift (+17.2x)")
    ax.axvspan(15, 30, alpha=0.08, color="#3498db", label="Phase 3: RL Healing")
    ax.axvspan(24, 26, alpha=0.20, color="#27ae60", label="Days 24-26: GREEN gate")
    ax.set_xlabel("Simulation Day", fontsize=9.5)
    ax.set_ylabel("PSI (EMA smoothed)", fontsize=9.5)
    ax.set_title(PSI_SCHEMATIC_TITLE, fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=-0.2)
    fig.tight_layout()
    return fig


def fig_adaptation_curve():
    """
    FIX-3: Adaptation recovery curve using real finetune_fixed_results.json values.
    Shows 4-class macro-F1 and binary F1 separately, with 0.75 theoretical ceiling.
    Old version used fabricated values from the rejected variant.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # 4-class line — converges on 0.75 theoretical maximum
    ax.plot(FINETUNE_PCTS, FINETUNE_F1_4CLASS, "o-", color="#3498db", lw=2,
            ms=7, label="4-class macro-F1")
    # Binary line
    ax.plot(FINETUNE_PCTS, FINETUNE_F1_BINARY, "s--", color="#27ae60", lw=1.5,
            ms=6, label="Binary F1 (healthy vs fault)")
    # Theoretical ceiling
    ax.axhline(FINETUNE_THEO_MAX, color="#9b59b6", lw=1.4, ls=":",
               label=f"Theoretical max = {FINETUNE_THEO_MAX} (Class 1 absent)")
    # Baseline
    ax.axhline(0.3116, color="#e74c3c", ls="--", lw=1.2,
               label="Unadapted baseline (0.3116)")
    # 10% recommendation marker
    ax.axvline(10, color="#f39c12", ls="--", lw=1.2, label="10% threshold")

    ax.annotate(
        f"4-class: 0.7440\n(99.2% of max)",
        xy=(10, 0.7440), xytext=(18, 0.60),
        arrowprops=dict(arrowstyle="->", color="#3498db"),
        fontsize=8.5, color="#3498db"
    )
    ax.annotate(
        "Binary: 0.992",
        xy=(10, 0.992), xytext=(18, 0.87),
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        fontsize=8.5, color="#27ae60"
    )

    ax.set_xlabel("Target-building data used (%)", fontsize=9.5)
    ax.set_ylabel("F1 Score", fontsize=9.5)
    ax.set_title("Adaptation Recovery Curve", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.2, 1.05)
    ax.set_xlim(-1, 52)
    _style(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── PAGE 1: SYSTEM OVERVIEW ─────────────────────────────────────────────────

# ── COMPACT FIGURE FUNCTIONS — for single-frame landscape screenshots ─────────

def fig_psi_bar_compact():
    sensors = list(PSI_VALUES.keys())
    values  = list(PSI_VALUES.values())
    clrs    = [PSI_GATE_COLOURS[psi_gate(v)] for v in values]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.barh(sensors, values, color=clrs, edgecolor="white", height=0.55)
    ax.axvline(PSI_THRESHOLDS["AMBER"], color="#f39c12", lw=1.2, ls="--")
    ax.axvline(PSI_THRESHOLDS["RED"],   color="#e74c3c", lw=1.2, ls="--")
    for bar, val in zip(bars, values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5)
    _style(ax, title="PSI — All 11 Sensors", xlabel="PSI Value")
    ax.set_xlim(0, max(values) * 1.22)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.8)
    return fig


def fig_adaptation_compact():
    pcts      = [0, 5, 10, 15, 20, 30, 50]
    f1_4class = [0.3116, 0.7395, 0.7440, 0.7468, 0.7474, 0.7485, 0.7498]
    f1_binary = [0.4154, 0.9860, 0.9920, 0.9958, 0.9965, 0.9980, 0.9997]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.plot(pcts, f1_4class, "o-", color="#3498db", lw=1.8, ms=5, label="4-class F1")
    ax.plot(pcts, f1_binary, "s--", color="#27ae60", lw=1.5, ms=4, label="Binary F1")
    ax.axhline(0.75, color="#9b59b6", lw=1.2, ls=":", label="Max 0.75")
    ax.axhline(0.3116, color="#e74c3c", ls="--", lw=1.0, label="Baseline")
    ax.axvline(10, color="#f39c12", ls="--", lw=1.0)
    ax.annotate("0.7440\n(99.2%)", xy=(10, 0.744), xytext=(22, 0.58),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=0.8),
                fontsize=7.5, color="#3498db")
    ax.set_xlabel("Target data used (%)", fontsize=8.5)
    ax.set_ylabel("F1 Score", fontsize=8.5)
    ax.set_title("Adaptation Recovery Curve", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_ylim(0.2, 1.05)
    _style(ax); fig.tight_layout(pad=0.8)
    return fig


def fig_fault_bar_compact(df):
    counts = df["Predicted_Class"].value_counts().reindex(
        LABEL_MAP.values(), fill_value=0)
    total = counts.sum() or 1
    clrs  = [FAULT_CONFIG[k][1] for k in counts.index]
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    bars = ax.barh(counts.index, counts.values, color=clrs, height=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val/total*100:.0f}%", va="center", fontsize=8)
    _style(ax, title="Predicted Fault Distribution", xlabel="Records")
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.8)
    return fig


def fig_confusion_compact(df):
    from sklearn.metrics import confusion_matrix as sk_cm
    labels = list(LABEL_MAP.keys())
    names  = ["Healthy", "Sensor\nBias", "Valve\nFault", "Damper\nFault"]
    cm     = sk_cm(df["Actual_Label"], df["Predicted_Label"], labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    annot  = [[f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)" for j in range(4)] for i in range(4)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.3, cbar_kws={"label": "Row %", "shrink": 0.8},
                vmin=0, vmax=100)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
    ax.set_title("Multi-Fault Confusion Matrix", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout(pad=0.8)
    return fig


def fig_learning_compact():
    props  = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    f1_lc  = [0.5463, 0.5473, 0.5155, 0.5348, 0.5472, 0.5317,
              0.5309, 0.5306, 0.4983, 0.5224, 0.5592]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(props, f1_lc, "s-", color="#9b59b6", lw=1.8, ms=5)
    ax.axhline(0.331, color="#e74c3c", ls="--", lw=1.1, label="Zero-data (0.331)")
    ax.fill_between(props, 0.490, 0.570, alpha=0.12, color="#9b59b6", label="Flat zone")
    ax.set_xlabel("Training data (%)", fontsize=8.5)
    ax.set_ylabel("Cross-bldg F1", fontsize=8.5)
    ax.set_title("Learning Curve — Gap is Structural", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5); _style(ax)
    fig.tight_layout(pad=0.8)
    return fig


def fig_psi_sim_compact():
    np.random.seed(42)
    days    = np.linspace(0, 30, 43200)
    fan_psi = np.where(
        days < 5, np.random.uniform(0, 0.05, 43200),
        np.where(days < 15,
                 (days - 5) / 10 * 15 + np.random.normal(0, 0.3, 43200),
                 np.maximum(0, 14 - (days - 15) * 0.5 +
                            np.random.normal(0, 0.8, 43200)))
    )
    fan_smooth = pd.Series(fan_psi).ewm(alpha=0.001).mean().values
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(days, fan_smooth, color="#e74c3c", lw=1.0, label="Fan Speed PSI")
    ax.axhline(PSI_THRESHOLDS["AMBER"], color="#f39c12", lw=1.2, ls="--",
               label=f"AMBER ({PSI_THRESHOLDS['AMBER']:.2f})")
    ax.axhline(PSI_THRESHOLDS["RED"], color="#e74c3c", lw=1.2, ls="--",
               label=f"RED ({PSI_THRESHOLDS['RED']:.2f})")
    ax.axvspan(0,  5,  alpha=0.08, color="#27ae60")
    ax.axvspan(5,  15, alpha=0.08, color="#e74c3c")
    ax.axvspan(15, 30, alpha=0.08, color="#3498db")
    ax.axvspan(24, 26, alpha=0.22, color="#27ae60", label="Days 24-26 GREEN")
    ax.set_xlabel("Simulation Day", fontsize=8.5)
    ax.set_ylabel("PSI (EMA)", fontsize=8.5)
    ax.set_title(PSI_SCHEMATIC_TITLE, fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, 30); ax.set_ylim(bottom=-0.2)
    fig.tight_layout(pad=0.8)
    return fig

def page_overview(df, fdf):
    m = THESIS_METRICS
    st.markdown("### 🏢 System Overview")
    st.caption("AHU Multi-Fault Detection · Primary Model · run_20260409_022716_ASHRAE_LBNL")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("In-sample F1",          f"{m['f1_insample']:.4f}",
              f"CI [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")
    c2.metric("Cross-building F1",     f"{m['f1_crossbldg']:.3f}",
              f"-{m['gap_pp']:.2f} pp", delta_color="inverse")
    c3.metric("Fan Speed PSI",         f"{m['fan_psi']:.4f}", "RED gate")
    c4.metric("OA Temp PSI",           f"{m['oa_psi']:.4f}",  "RED gate")
    c5.metric("Binary F1 (10% adapt)", f"{m['binary_adapt']:.3f}", "+58.1 pp recovery")

    st.markdown("---")
    col_left, col_right = st.columns([0.54, 0.46])

    with col_left:
        st.markdown("**PSI Deployment Gate — All 11 Sensors**")
        st.caption(f"{PSI_BANDS_TEXT} (Yurdakul & Naranjo, 2019)")
        psi_cards = []
        for sensor, psi_val in PSI_VALUES.items():
            gate = psi_gate(psi_val)
            if gate == "RED":
                bg, border, dot, label = "#3D0000", "#E74C3C", "&#128308;", "RED"
            elif gate == "AMBER":
                bg, border, dot, label = "#3D2B00", "#F39C12", "&#128993;", "AMBER"
            else:
                bg, border, dot, label = "#003D1A", "#27AE60", "&#128994;", "GREEN"
            psi_cards.append(
                f"<div style='background:{bg};border:2px solid {border};"
                f"border-radius:6px;padding:6px 8px;text-align:center;flex:1 1 75px;"
                f"min-width:68px;max-width:105px;'>"
                f"<div style='font-size:13px;'>{dot}</div>"
                f"<div style='font-weight:700;font-size:10px;color:#FFF;margin:2px 0 1px;"
                f"word-break:break-all;'>{sensor}</div>"
                f"<div style='font-size:10px;color:{border};font-weight:600;'>{label}</div>"
                f"<div style='font-size:9.5px;color:#CCC;'>{psi_val:.4f}</div>"
                f"</div>"
            )
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;'>"
            + "".join(psi_cards) + "</div>", unsafe_allow_html=True
        )
        _show(fig_psi_bar_compact())
        st.caption(
            f"Source: psi_top_features in results_summary.json. "
            f"Fan Speed ({m['fan_psi']:.4f}) and OA Temp ({m['oa_psi']:.4f}) "
            "are the primary structural shift drivers."
        )

    with col_right:
        k = kpis(fdf)
        st.markdown("**Live Data KPIs**")
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Total Records", f"{k['total']:,}")
        kc2.metric("Fault Rate",    f"{k['fault_rate']}%")
        kc3.metric("Healthy",       f"{k['healthy_pct']}%")
        kc4.metric("Top Fault",      k['top_fault'])

        st.markdown("**Predicted Fault Distribution**")
        if not fdf.empty:
            _show(fig_fault_bar_compact(fdf))
        else:
            st.info("No data in selected date range.")

        st.markdown("**Adaptation Recovery Curve**")
        _show(fig_adaptation_compact())
        st.caption(
            "4-class F1 = 0.7440 at 10% (99.2% of theoretical max 0.75). "
            "Source: finetune_fixed_results.json."
        )



def page_fault_analysis(df, fdf):
    st.markdown("### 🔍 Fault Analysis")
    st.caption("Confusion matrix · Per-class metrics · Cross-building comparison")

    if fdf.empty or "Actual_Label" not in fdf.columns:
        st.warning("Actual labels not available in current dataset.")
        st.info(
            f"In-sample macro-F1: **{THESIS_METRICS['f1_insample']:.4f}** "
            f"CI [{THESIS_METRICS['ci_lo']:.4f}, {THESIS_METRICS['ci_hi']:.4f}]\n\n"
            f"Cross-building macro-F1: **{THESIS_METRICS['f1_crossbldg']:.3f}** "
            f"(gap = {THESIS_METRICS['gap_pp']:.2f} pp)\n\n"
            "Per-class in-sample (Table 4.3): Healthy 0.9828 · Sensor Bias 0.9973 · "
            "Valve 1.0000 · Damper 1.0000"
        )
        return

    col_left, col_right = st.columns([0.52, 0.48])

    with col_left:
        st.markdown("**Multi-Fault Confusion Matrix**")
        _show(fig_confusion_compact(fdf))
        st.markdown("**Per-Class Metrics**")
        metrics_df = per_class_metrics(fdf)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True, height=180)

    with col_right:
        st.markdown("**Cross-Building Generalisation Summary**")
        compare = pd.DataFrame({
            "Deployment Scenario": [
                "In sample (training buildings)",
                "Cross-building (withheld buildings)",
                "After 10% adaptation",
                "Real-world average (Seoul/Cork)",
            ],
            "Macro F1":   [0.9923, 0.331, 0.7440, 0.084],
            "Binary F1":  [0.9990, 0.415, 0.9920, "N/A"],
            "Gap (pp)":   ["—", "66.13", "25.0", ">90"],
        })
        st.dataframe(compare, hide_index=True, use_container_width=True, height=175)
        st.caption("Source: Self-created from verified JSON experiment files.")
        st.markdown("**Learning Curve — Why More Data Does Not Help**")
        _show(fig_learning_compact())
        st.caption(
            "F1 stays flat 0.4983–0.5592 across all 11 training proportions. "
            "Source: learning_curve_results.json."
        )




# ── Pre-computed SHAP values from shap_values.json (run_20260409_022716) ────
# Mean |SHAP| per raw sensor — computed on 500 cross-building test samples
PRECOMPUTED_SHAP = [
    {"rank": r["rank"], "feature": r["feature"], "mean_abs_shap": round(r["mean_abs_shap"], 4)}
    for r in json.loads(SHAP_RESULTS_PATH.read_text(encoding="utf-8"))
]
TOP_SHAP_FEATURE = PRECOMPUTED_SHAP[0]["feature"]
TOP_SHAP_VALUE   = PRECOMPUTED_SHAP[0]["mean_abs_shap"]

SENSOR_FULL_NAMES = {
    "SA_Temp":    "AHU: Supply Air Temperature",
    "OA_Temp":    "AHU: Outdoor Air Temperature",
    "MA_Temp":    "AHU: Mixed Air Temperature",
    "RA_Temp":    "AHU: Return Air Temperature",
    "Fan_Status": "AHU: Supply Air Fan Status",
    "Fan_Speed":  "AHU: Supply Air Fan Speed Control Signal",
    "OA_Damper":  "AHU: Outdoor Air Damper Control Signal",
    "RA_Damper":  "AHU: Return Air Damper Control Signal",
    "CC_Valve":   "AHU: Cooling Coil Valve Control Signal",
    "HC_Valve":   "AHU: Heating Coil Valve Control Signal",
    "Occupancy":  "Occupancy Mode Indicator",
}


def feature_psi_gate(feature: str) -> str:
    """PSI gate of the raw sensor channel a feature is built from ("—" if derived)."""
    base = re.sub(r"_r[ms]\d+$", "", feature)
    short = {v: k for k, v in SENSOR_FULL_NAMES.items()}.get(base)
    return psi_gate(PSI_VALUES[short]) if short else "—"

# ── Top-12 impurity importance, Table 4.4 (canonical run_20260409_022716, 70 features) ─
PRECOMPUTED_IMPORTANCE = [
    {"rank": 1,  "feature": "AHU: Return Air Damper Control Signal",        "importance": 0.0932},
    {"rank": 2,  "feature": "AHU: Outdoor Air Temperature_rm60",            "importance": 0.0607},
    {"rank": 3,  "feature": "AHU: Outdoor Air Temperature_rm30",            "importance": 0.0526},
    {"rank": 4,  "feature": "AHU: Outdoor Air Temperature_rm10",            "importance": 0.0492},
    {"rank": 5,  "feature": "AHU: Outdoor Air Temperature",                 "importance": 0.0461},
    {"rank": 6,  "feature": "AHU: Supply Air Temperature_rm30",             "importance": 0.0323},
    {"rank": 7,  "feature": "Temp_supply_return_diff",                      "importance": 0.0314},
    {"rank": 8,  "feature": "AHU: Supply Air Temperature_rm60",             "importance": 0.0311},
    {"rank": 9,  "feature": "AHU: Heating Coil Valve Control Signal_rs10",  "importance": 0.0275},
    {"rank": 10, "feature": "Temp_outdoor_supply_diff",                     "importance": 0.0255},
    {"rank": 11, "feature": "AHU: Supply Air Temperature",                  "importance": 0.0237},
    {"rank": 12, "feature": "AHU: Supply Air Temperature_rm10",             "importance": 0.0234},
]

def fig_shap_bar_precomputed():
    """Bar chart of pre-computed SHAP values from shap_values.json."""
    names  = [r["feature"].replace("AHU: ", "") for r in PRECOMPUTED_SHAP]
    vals   = [r["mean_abs_shap"] for r in PRECOMPUTED_SHAP]
    clrs   = ["#e74c3c" if v >= 0.05 else "#f39c12" if v >= 0.02 else "#3498db"
              for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names[::-1], vals[::-1], color=clrs[::-1],
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)
    _style(ax,
           title="Mean |SHAP| per Raw Sensor Channel — Cross-Building Test Set",
           xlabel="Mean |SHAP| value")
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    return fig


def fig_importance_bar_precomputed():
    """Bar chart of top-12 impurity importance from Table 4.4."""
    names  = [r["feature"].replace("AHU: ", "") for r in PRECOMPUTED_IMPORTANCE]
    vals   = [r["importance"] for r in PRECOMPUTED_IMPORTANCE]
    clrs   = (["#e74c3c"] * 3 + ["#f39c12"] * 4 + ["#3498db"] * 5)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], vals[::-1], color=clrs[::-1],
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)
    _style(ax,
           title="Top 12 Feature Importance (Impurity) — 70-Feature Model (Table 4.4)",
           xlabel="Gini Importance")
    ax.set_xlim(0, max(vals) * 1.3)
    fig.tight_layout()
    return fig


def page_explainability(df, fdf):
    st.title("Explainability — SHAP Analysis")
    st.caption(
        "SHAP TreeExplainer (Lundberg & Lee, 2017) · "
        "Source: results/shap_values.json + Table 4.4 (canonical run_20260409_022716)"
    )

    st.info(
        "**Top feature by impurity importance (70-feature model, Table 4.4):** "
        "Return Air Damper Control Signal — importance = **0.0932**. "
        "This is a BMS command, not a physical measurement. "
        "It encodes each building's commissioned outdoor-air fraction strategy — "
        "the primary mechanism of cross-building failure.\n\n"
        "**Top raw sensor by SHAP (shap_values.json):** "
        f"{TOP_SHAP_FEATURE.replace('AHU: ', '')} — mean |SHAP| = **{TOP_SHAP_VALUE:.4f}**. "
        "All five OA temperature features (raw + rm10/30/60) rank in the top 5 by impurity."
    )

    # Two-column layout: fits in one screenshot frame
    ex_left, ex_right = st.columns([0.52, 0.48])

    with ex_left:
        st.markdown("**Impurity Importance — Top 12 (Table 4.4)**")
        st.caption("Source: Table 4.4 (canonical run_20260409_022716) · Top 12 = 49.7% of total importance")
        _show(fig_importance_bar_precomputed())
        imp_df = pd.DataFrame([{**r, "psi_gate": feature_psi_gate(r["feature"])}
                               for r in PRECOMPUTED_IMPORTANCE])
        imp_df.columns = ["Rank", "Feature", "Gini Importance", "PSI Gate"]
        st.dataframe(imp_df, hide_index=True, use_container_width=True, height=390)
        st.caption(
            "Every top-5 feature sits on a RED-gate sensor channel — "
            "the model relies most on the channels that shift most between buildings."
        )

    with ex_right:
        st.markdown("**Mean |SHAP| per Raw Sensor (shap_values.json)**")
        st.caption("Computed on 500 cross-building test samples using SHAP TreeExplainer")
        _show(fig_shap_bar_precomputed())

        st.markdown("**PSI vs SHAP Attribution — Why the Gap Exists**")
        psi_shap_data = []
        shap_lookup = {r["feature"]: r["mean_abs_shap"] for r in PRECOMPUTED_SHAP}
        for short, psi_val in PSI_VALUES.items():
            full = SENSOR_FULL_NAMES.get(short, short)
            shap_val = shap_lookup.get(full, 0.0)
            psi_shap_data.append({
                "Sensor":       short,
                "PSI":          round(psi_val, 4),
                "Mean |SHAP|":  round(shap_val, 4),
                "Gate": {"RED": "🔴 RED", "AMBER": "🟡 AMBER",
                         "GREEN": "🟢 GREEN"}[psi_gate(psi_val)],
            })
        st.dataframe(pd.DataFrame(psi_shap_data),
                     hide_index=True, use_container_width=True, height=388)
        st.caption(
            "Fan Speed (2.63 🔴) and OA Temp (2.45 🔴): highest shift AND highest "
            "SHAP reliance. Fan Status (0.01 🟢) and Occupancy (0.0002 🟢): transfer safely."
        )



def page_cost_calculator(df, fdf):
    st.markdown("### 💶 Cost Calculator")
    st.caption("Probability-weighted energy cost model · EUR estimates")

    col_inputs, col_results = st.columns([0.42, 0.58])

    with col_inputs:
        st.markdown("**AHU Parameters**")
        capacity    = st.slider("AHU capacity (kW)", 50, 2000, 500, 50)
        elec_price  = st.slider("Electricity (EUR/kWh)", 0.05, 0.50, 0.25, 0.01)
        hours_day   = st.slider("Operating hours/day", 8, 24, 12, 1)
        labour_rate = st.slider("Labour rate (EUR/h)", 50, 150, 85, 5)
        st.markdown("**Fault Scenario**")
        selected_cls = st.selectbox("Select fault class to evaluate:",
                                    list(LABEL_MAP.values()), index=1)
        fault_prob_default = {
            "Healthy": 0.10, "Sensor Bias": 0.65,
            "Valve Fault": 0.80, "Damper Fault": 0.70,
        }
        fault_prob = st.slider("Model prediction probability", 0.0, 1.0,
                               fault_prob_default.get(selected_cls, 0.65), 0.01)

    costs  = compute_cost(selected_cls, fault_prob, capacity, elec_price, hours_day, labour_rate)
    params = costs["params"]

    with col_results:
        st.markdown("**Cost Summary**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Do nothing / month",  f"EUR {costs['total_do_nothing']:,.0f}")
        cc2.metric("Repair now (one-off)", f"EUR {costs['repair_now_total']:,.0f}")
        cc3.metric("Net monthly saving",  f"EUR {costs['net_monthly_saving']:,.0f}")
        if costs["payback_months"]:
            st.success(f"**Payback period: {costs['payback_months']} months**")
        st.markdown("**Detailed Breakdown**")
        breakdown = pd.DataFrame({
            "Cost Component": [
                "Base energy / month", "Energy penalty (fault) / month",
                "Downtime risk / month", "Emergency parts amortised / month",
                "Emergency labour amortised / month", "Total — do nothing / month",
                "Scheduled repair (one-off)", "Repair amortised / month",
                "Net monthly saving",
            ],
            "EUR": [
                costs["base_energy_monthly"], costs["energy_penalty"],
                costs["downtime_cost"], costs["emergency_parts_amort"],
                costs["emergency_labour"], costs["total_do_nothing"],
                costs["repair_now_total"], costs["repair_now_monthly"],
                costs["net_monthly_saving"],
            ]
        })
        breakdown["EUR"] = breakdown["EUR"].apply(lambda x: f"EUR {x:,.2f}")
        st.dataframe(breakdown, hide_index=True, use_container_width=True, height=295)
        st.markdown("**SHAP-Linked Interpretation**")
        st.info(params["interpretation"])
        with st.expander("Cost parameter sources"):
            st.caption(params["source"])



def page_self_healing_loop():
    st.markdown("### 🤖 Self-Healing Loop")
    st.caption(
        "Autonomous Cyber-Physical Extension · "
        "Online RL + CBF-QP Safety Filter · 30-Day Simulation"
    )

    m = THESIS_METRICS

    st.success(
        f"✅ **SAFETY RESULT: {m['cbf_violations']} CBF violations** across "
        f"**{m['sim_minutes']:,} continuous minutes** (30 days) "
        f"and **{m['real_timesteps']:,} real BMS timesteps** (5 datasets). "
        "ASHRAE 62.1 OA fraction ≥ 10%, static pressure [100, 375] Pa. "
        "**Mathematical guarantee from CBF-QP filter.**"
    )

    z1, z2, z3, z4 = st.columns(4)
    z1.metric("ZSSC-1: Binary F1 > 0.850", "0.336 (Day 29)", "FAIL gate-induced", delta_color="off")
    z2.metric("ZSSC-2: 4-class F1 > 0.350", "0.343 (Day 29)", "borderline", delta_color="off")
    z3.metric("ZSSC-3: PSI < 0.500",        "0.331 avg",      "near threshold", delta_color="off")
    z4.metric("ZSSC-4: CBF = 0",            "0 / 43,200 min", "PASS ✅")

    # FIX-9: surrogate disclaimer on ZSSC
    st.warning(
        "**Important:** ZSSC metrics are from the 30-day simulation, which uses a "
        "calibrated surrogate classifier (MockFDD, Section 5.5), not rf_model.pkl. "
        "The surrogate is parameterised to reproduce the measured anchors of this study "
        "(approx. 0.99 under zero shift, 0.415 binary F1 at full shift). "
        "F1 figures on this page describe loop behaviour, not trained model performance."
    )

    st.info(
        "**Why F1 = 0.336 is NOT a failure:** PSI gate correctly paused the classifier "
        "during high-drift periods. During Days 24–26 (GREEN gate): "
        f"**Binary F1 = {m['green_binary']:.3f}, 4-class F1 = {m['green_4class']:.3f}** "
        f"— both exceed the thesis cross-building baseline of {m['f1_crossbldg']:.3f}."
    )

    col_left, col_right = st.columns([0.50, 0.50])

    with col_left:
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Gate State",   "GREEN ✅")
        gc2.metric("Binary F1",    f"{m['green_binary']:.3f}", "+27.9pp vs baseline")
        gc3.metric("4-class F1",   f"{m['green_4class']:.3f}",
                   f"+{m['green_4class'] - m['f1_crossbldg']:.3f} vs baseline")

        st.markdown(f"**{PSI_SCHEMATIC_TITLE}**")
        _show(fig_psi_sim_compact())
        st.caption(
            f"{PSI_SCHEMATIC_TITLE}. "
            "Phase 1 (0–5d): clean. Phase 2 (5–15d): +17.2× shift. "
            "Phase 3 (15–30d): RL healing. Days 24–26: GREEN gate (highlighted)."
        )

        st.markdown("**Architectural Remedies for Full Convergence**")
        r1, r2, r3 = st.columns(3)
        r1.markdown("**R1: Ref Reset**\nExp age-decay λ=0.01 after each RL actuation.")
        r2.markdown("**R2: Action Smooth**\nPenalty −2·‖Δa‖² damps limit cycles.")
        r3.markdown("**R3: Variance Decay**\nLog-σ ceiling 0→−2 over first 5d Phase 3.")

    with col_right:
        st.markdown("**Four-Layer Architecture**")
        t1, t2, t3, t4 = st.tabs(["BMS-MAE", "PSI Guardian", "RL Controller", "Validation"])
        with t1:
            st.markdown(
                "**Thermodynamic Invariance**\n\n"
                "• Pre-trained on Cork: 194,048 unlabelled rows\n"
                "• 75% masking → learns thermodynamic laws\n"
                "• Building-invariant latent representations\n"
                "• Zero-shot cross-building transfer"
            )
        with t2:
            st.markdown(
                "**Streaming Shift Detection**\n\n"
                "• Reference: 168h rolling · Deployment: 24h rolling\n"
                "• Update: every 60 min · EMA α = 0.30\n"
                f"• AMBER = {PSI_THRESHOLDS['AMBER']:.2f} · RED = {PSI_THRESHOLDS['RED']:.2f}\n"
                "• Pauses FDD classifier when gate = RED"
            )
        with t3:
            st.markdown(
                "**SHAP-Guided RL**\n\n"
                "• REINFORCE Actor-Critic · 14-dim state · 3-dim action\n"
                "• R_t = 10·ΔPSI − 100·comfort² + 1·ΔE − 2·‖Δa‖²\n"
                "• CBF-QP filter: SLSQP · 6 constraints · 100% convergence\n"
                "• Online updates every 24 transitions"
            )
        with t4:
            results = pd.DataFrame({
                "Metric": [
                    "CBF violations (real BMS)",
                    "CBF violations (simulation)",
                    "SLSQP convergence",
                    "Days 24-26 binary F1",
                    "Days 24-26 4-class F1",
                    "Verification",
                    "Adversarial trials",
                ],
                "Value": [
                    f"0 / {m['real_timesteps']:,} timesteps",
                    "0 / 43,200 min",
                    "100% (all calls)",
                    f"{m['green_binary']:.3f}",
                    f"{m['green_4class']:.3f}",
                    "4/4 PASS in 1.37s",
                    "0 / 600 trials",
                ],
            })
            st.dataframe(results, hide_index=True, use_container_width=True, height=230)

    st.caption(
        "Self-Healing Loop · layer4_validation_harness.py · "
        "Harshil Patel · M.Sc. Smart Building Technologies · bbw Hochschule Berlin"
    )



st.set_page_config(
    page_title="DriftGuard — AHU Fault Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## 🛡️ DriftGuard")
    st.caption("AHU Multi-Fault Detection")
    st.markdown(
        "<div style='font-size:10px;color:#888;line-height:1.6;'>"
        "M.Sc. Smart Building Technologies<br>bbw Hochschule Berlin<br>Harshil Patel"
        "</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:9px;color:#3498db;font-style:italic;'>"
        "Deploy with confidence. Adapt automatically. Explain every alert."
        "</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Primary model metrics:**")
    st.caption(f"In-sample F1: **{THESIS_METRICS['f1_insample']:.4f}**  CI [{THESIS_METRICS['ci_lo']:.4f}, {THESIS_METRICS['ci_hi']:.4f}]")
    st.caption(f"Cross-building F1: **{THESIS_METRICS['f1_crossbldg']:.3f}**  Gap: **{THESIS_METRICS['gap_pp']:.2f} pp**")
    st.caption(f"Binary F1 (10% adapt): **{THESIS_METRICS['binary_adapt']:.3f}**")
    st.caption(f"CBF: **0** / {THESIS_METRICS['real_timesteps']:,} real + {THESIS_METRICS['sim_minutes']:,} sim")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "page", label_visibility="collapsed",
        options=[
            "System Overview",
            "Fault Analysis",
            "Explainability (SHAP)",
            "Cost Calculator",
            "Self-Healing Loop",
        ],
    )

    if page != "Self-Healing Loop":
        st.divider()
        st.markdown("### Filters")

# ── Page 6 needs no data files ───────────────────────────────────────────────
if page == "Self-Healing Loop":
    page_self_healing_loop()
    st.stop()

# ── Check required files ─────────────────────────────────────────────────────
missing = [str(p) for p in [DATA_PATH, SHAP_RESULTS_PATH] if not p.exists()]
if missing:
    st.error("Missing required data files. Run the data preparation pipeline first.")
    for f in missing:
        st.code(f)
    st.info(
        "**You can still view the Self-Healing Loop page** — "
        "it does not require data files. Select it from the sidebar."
    )
    st.stop()

# ── Load resources ────────────────────────────────────────────────────────────
try:
    df = load_data()
except Exception as e:
    st.error(f"Resource loading failed: {e}")
    st.stop()

# ── Date and fault filters ────────────────────────────────────────────────────
with st.sidebar:
    min_d = df["Timestamp"].min().date()
    max_d = df["Timestamp"].max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
    end   = c2.date_input("To",   value=max_d, min_value=min_d, max_value=max_d)
    if start > end:
        st.error("Start must be before end.")

    all_faults = sorted(df["Predicted_Class"].dropna().unique().tolist())
    sel_faults = st.multiselect("Fault classes", all_faults, default=all_faults)


fdf = filter_df(df, start, end, sel_faults)

# ── Route ─────────────────────────────────────────────────────────────────────
if   page == "System Overview":       page_overview(df, fdf)
elif page == "Fault Analysis":        page_fault_analysis(df, fdf)
elif page == "Explainability (SHAP)": page_explainability(df, fdf)
elif page == "Cost Calculator":       page_cost_calculator(df, fdf)
