"""
Step 4 — PSI Gate
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Computes Population Stability Index (PSI) between the training reference
distribution and a new building's data, per feature column.

PSI thresholds (thesis-calibrated):
  PSI < 0.20  → GREEN  — distribution stable, deploy immediately
  PSI < 0.50  → AMBER  — moderate shift, collect adaptation data
  PSI >= 0.50 → RED    — extreme drift, adaptation required

Usage:
  python psi_gate.py <equipment> <path/to/new_building.csv>

  python psi_gate.py ahu   data\MZVAV-2-2.csv
  python psi_gate.py fcu   data\fcu\FCU_FaultFree.csv
  python psi_gate.py boiler data\boiler\BoilerPlant.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODELS  = r"C:\Users\hrslp\Desktop\thesis\models\registry"
EVAL_CAP     = 15_000
ADAPT_CAP    = 50_000
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
EXCLUDE_COLS = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
    "actual_class", "actual_label", "predicted_class", "predicted_label",
}

PSI_GREEN = 0.20   # below → GREEN: deploy immediately
PSI_AMBER = 0.50   # below → AMBER: collect adaptation data; above → RED: extreme shift
N_BINS    = 10     # buckets for PSI calculation

# Files containing these strings are pipeline/output files — skip them
SKIP_FILENAME_SUBSTRINGS = ["predictions", "feature_sample", "chiller_feature"]

# FCU column rename map (native → AHU-equivalent, matching what the model trained on)
FCU_COL_MAP = {
    "FCU_CVLV": "AHU: Cooling Coil Valve Control Signal",
    "FCU_HVLV": "AHU: Heating Coil Valve Control Signal",
    "FCU_MAT":  "AHU: Mixed Air Temperature",
    "FCU_DMPR": "AHU: Outdoor Air Damper Control Signal",
    "FCU_OAT":  "AHU: Outdoor Air Temperature",
    "FCU_RAT":  "AHU: Return Air Temperature",
    "FCU_SPD":  "AHU: Supply Air Fan Speed Control Signal",
    "FCU_DAT":  "AHU: Supply Air Temperature",
}

BOILER_STRUCTURAL_NAN_COLS = ["PM_POW_1", "PM_POW_2"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_load(path: str, nrows: int = EVAL_CAP) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    return df


def load_registry(equipment: str) -> tuple:
    """Load feature_cols and training_reference for an equipment type."""
    base = os.path.join(BASE_MODELS, equipment)
    with open(os.path.join(base, "feature_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    with open(os.path.join(base, "training_reference.pkl"), "rb") as f:
        ref_df = pickle.load(f)
    return feat_cols, ref_df


def preprocess(df: pd.DataFrame, equipment: str, feat_cols: list) -> pd.DataFrame:
    """Apply equipment-specific preprocessing and select feature columns."""
    if equipment == "fcu":
        df = df.rename(columns=FCU_COL_MAP)
    if equipment == "boiler":
        for col in BOILER_STRUCTURAL_NAN_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    available = [c for c in feat_cols if c in df.columns]
    return df[available].dropna()


def psi_single(ref_series: pd.Series, new_series: pd.Series, n_bins: int = N_BINS) -> float:
    """
    Compute PSI for one feature column.
    Uses quantile-based bins from the reference distribution.
    Clips proportions away from 0 to avoid log(0).
    """
    # Build bin edges from reference (avoid duplicate edges at extremes)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref_series.dropna(), quantiles))

    if len(bin_edges) < 2:
        return 0.0   # constant feature — no drift possible

    # Extend edges to cover new data outside training range
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    ref_counts = np.histogram(ref_series.dropna(), bins=bin_edges)[0]
    new_counts = np.histogram(new_series.dropna(), bins=bin_edges)[0]

    ref_pct = ref_counts / ref_counts.sum()
    new_pct = new_counts / new_counts.sum()

    # Clip to avoid log(0) — standard practice
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    new_pct = np.clip(new_pct, eps, None)

    psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return float(psi)


def traffic_light(psi: float) -> str:
    if psi < PSI_GREEN:
        return "🟢 GREEN"
    elif psi < PSI_AMBER:
        return "🟡 AMBER"
    else:
        return "🔴 RED"


# ── Main PSI computation ──────────────────────────────────────────────────────

def compute_psi(equipment: str, new_file: str) -> dict:
    """
    Full PSI gate for one file.
    Returns dict with: equipment, file, overall_psi, gate, per_feature_psi, n_features_red.
    """
    print(f"\n{'='*65}")
    print(f"PSI GATE  |  Equipment: {equipment.upper()}  |  File: {os.path.basename(new_file)}")
    print(f"{'='*65}")

    # Load registry
    try:
        feat_cols, ref_df = load_registry(equipment)
    except FileNotFoundError:
        print(f"  ERROR: No registry model found for '{equipment}'.")
        print(f"  Run specialist_trainer.py {equipment} first.")
        return {}

    # Load new file
    new_df = safe_load(new_file, nrows=EVAL_CAP)
    new_df = preprocess(new_df, equipment, feat_cols)

    if len(new_df) == 0:
        print("  ERROR: No usable rows after preprocessing.")
        return {}

    # Align reference to same columns
    available_cols = [c for c in feat_cols if c in new_df.columns and c in ref_df.columns]
    if not available_cols:
        print("  ERROR: No overlapping columns between reference and new file.")
        return {}

    ref_aligned = ref_df[available_cols]
    new_aligned = new_df[available_cols]

    print(f"  Reference rows : {len(ref_aligned):,}")
    print(f"  New file rows  : {len(new_aligned):,}")
    print(f"  Features       : {len(available_cols)}")
    print()

    # Compute PSI per feature
    per_feature = {}
    for col in available_cols:
        psi_val = psi_single(ref_aligned[col], new_aligned[col])
        per_feature[col] = psi_val

    # Overall PSI = mean across features
    overall_psi = float(np.mean(list(per_feature.values())))
    gate        = traffic_light(overall_psi)
    n_red       = sum(1 for v in per_feature.values() if v >= PSI_AMBER)
    n_amber     = sum(1 for v in per_feature.values() if PSI_GREEN <= v < PSI_AMBER)

    # Per-feature report
    print(f"  {'Feature':<48}  {'PSI':>6}  Status")
    print(f"  {'-'*48}  {'------':>6}  ------")
    for col, psi_val in sorted(per_feature.items(), key=lambda x: -x[1]):
        short = col[:47]
        print(f"  {short:<48}  {psi_val:>6.3f}  {traffic_light(psi_val)}")

    print()
    print(f"  ── OVERALL PSI : {overall_psi:.4f}  →  {gate}")
    print(f"  Features RED   : {n_red}/{len(available_cols)}")
    print(f"  Features AMBER : {n_amber}/{len(available_cols)}")
    print()

    # Recommendation
    if overall_psi < PSI_GREEN:
        rec = "GREEN — distribution stable. Deploy model immediately."
    elif overall_psi < PSI_AMBER:
        rec = "AMBER — moderate drift. Collect adaptation data before deployment."
    else:
        rec = "RED — extreme drift. Adaptation required (10% target data minimum)."
    print(f"  Recommendation : {rec}")

    return {
        "equipment":       equipment,
        "file":            os.path.basename(new_file),
        "overall_psi":     overall_psi,
        "gate":            gate,
        "per_feature_psi": per_feature,
        "n_features_red":  n_red,
        "n_features_amber": n_amber,
        "n_features":      len(available_cols),
        "recommendation":  rec,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python psi_gate.py <equipment> <path/to/file.csv>")
        print("  equipment: ahu | fcu | boiler")
        print()
        print("Examples:")
        print("  python psi_gate.py ahu   data\\MZVAV-2-2.csv")
        print("  python psi_gate.py fcu   data\\fcu\\FCU_FaultFree.csv")
        print("  python psi_gate.py boiler data\\boiler\\BoilerPlant.csv")
        sys.exit(1)

    equipment = sys.argv[1].lower()
    new_file  = sys.argv[2]

    if not os.path.isfile(new_file):
        print(f"ERROR: File not found: {new_file}")
        sys.exit(1)

    result = compute_psi(equipment, new_file)

    print(f"\n{'='*65}")
    print("PSI gate complete.")


if __name__ == "__main__":
    main()
