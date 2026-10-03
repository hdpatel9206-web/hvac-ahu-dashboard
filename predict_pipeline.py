"""
Step 5 — Universal Prediction Pipeline
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Takes any new CSV, runs the full registry pipeline:
  1. Skip filter  — rejects output/prediction files
  2. Equipment detection — filename then column signature
  3. PSI gate  — raw features vs training reference (before scaling)
  4. Scaler    — StandardScaler applied after PSI
  5. Prediction — fault class + per-class confidence

Universal fault classes:
  0 = Healthy
  1 = Sensor Bias
  2 = Valve / Coil / Heat Exchanger Fault
  3 = Control Fault

PSI thresholds:
  < 0.20  → GREEN  — deploy immediately
  < 0.50  → AMBER  — collect adaptation data
  >= 0.50 → RED    — extreme drift, adaptation required

Usage:
  python predict_pipeline.py path/to/file.csv
  python predict_pipeline.py path/to/folder/     # batch all CSVs
  python predict_pipeline.py path/to/file.csv --equipment ahu   # override detection
"""

import os
import sys
import glob
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODELS  = r"C:\Users\hrslp\Desktop\thesis\models\registry"
EVAL_CAP     = 15_000
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
EXCLUDE_COLS = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
    "actual_class", "actual_label", "predicted_class", "predicted_label",
}

PSI_GREEN = 0.20
PSI_AMBER = 0.50
N_BINS    = 10

FAULT_LABELS = {
    0: "Healthy",
    1: "Sensor Bias",
    2: "Valve / Coil / Heat Exchanger Fault",
    3: "Control Fault",
}

# Files to skip — output/prediction artifacts, not raw sensor data
SKIP_FILENAME_SUBSTRINGS = ["predictions", "feature_sample", "chiller_feature"]

# ── Equipment detection config ────────────────────────────────────────────────

FILENAME_SIGNATURES = {
    "boiler": ["boiler", "boilerplant", "hot_water"],
    "fcu":    ["fcu_", "fcu-"],
    "rtu":    ["rtu"],
    # AHU includes ASHRAE files AND SD-AHU fault scenario files
    "ahu":    [
        "mzvav", "szcav", "szvav", "ahu",
        "coi_", "damper_stuck", "oa_bias", "annual",
    ],
}

COLUMN_SIGNATURES = {
    "ahu": {
        "AHU: Cooling Coil Valve Control Signal",
        "AHU: Heating Coil Valve Control Signal",
        "AHU: Mixed Air Temperature",
        "AHU: Outdoor Air Damper Control Signal",
        "AHU: Outdoor Air Temperature",
        "AHU: Return Air Damper Control Signal",
        "AHU: Return Air Temperature",
        "AHU: Supply Air Fan Speed Control Signal",
        "AHU: Supply Air Fan Status",
        "AHU: Supply Air Temperature",
        "Occupancy Mode Indicator",
    },
    "fcu": {
        "FCU_CVLV", "FCU_HVLV", "FCU_MAT", "FCU_DMPR",
        "FCU_OAT", "FCU_RAT", "FCU_SPD", "FCU_DAT",
    },
    "boiler": {
        "OA_TEMP", "HWL_SW_TEMP", "HWL_RW_TEMP", "HWL_DP",
        "BOI_SW_TEMP_1", "BOI_RW_TEMP_1", "PM_SPD_1",
    },
    "rtu": {
        "RTU: Circuit 1 Discharge Pressure",
        "RTU: Circuit 1 Suction Pressure",
        "RTU: Supply Air Temperature",
        "RTU: Compressor 1 On/Off Status",
    },
}

# ── Equipment-specific column maps ────────────────────────────────────────────

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

BOILER_COL_MAP = {
    "OA_TEMP":       "Outdoor Air Temperature",
    "HWL_SW_TEMP":   "Hot Water Loop Supply Temperature",
    "HWL_RW_TEMP":   "Hot Water Loop Return Temperature",
    "HWL_DP":        "Hot Water Loop Differential Pressure",
    "BOI_SW_TEMP_1": "Boiler 1 Supply Water Temperature",
    "BOI_RW_TEMP_1": "Boiler 1 Return Water Temperature",
    "PM_SPD_1":      "Pump 1 Speed",
    "PM_POW_1":      "Pump 1 Power",
    "PM_POW_2":      "Pump 2 Power",
}

# SD-AHU column rename → AHU model feature names (best-effort mapping)
SDAHU_COL_MAP = {
    "CHWC_VLV":  "AHU: Cooling Coil Valve Control Signal",
    "OA_DMPR":   "AHU: Outdoor Air Damper Control Signal",
    "OA_TEMP":   "AHU: Outdoor Air Temperature",
    "RA_TEMP":   "AHU: Return Air Temperature",
    "RA_DMPR":   "AHU: Return Air Damper Control Signal",
    "SF_SPD":    "AHU: Supply Air Fan Speed Control Signal",
    "SF_CS":     "AHU: Supply Air Fan Status",
    "SA_TEMP":   "AHU: Supply Air Temperature",
    "MA_TEMP":   "AHU: Mixed Air Temperature",
    # Heating coil and Occupancy have no direct SD-AHU equivalent — will be missing
}

# ── Registry loader ───────────────────────────────────────────────────────────

def load_registry(equipment: str) -> dict:
    base = os.path.join(BASE_MODELS, equipment)
    required = ["rf_model.pkl", "scaler.pkl", "feature_cols.pkl", "training_reference.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(base, f)):
            raise FileNotFoundError(f"Missing {f} for equipment '{equipment}'")
    with open(os.path.join(base, "rf_model.pkl"),          "rb") as f: model     = pickle.load(f)
    with open(os.path.join(base, "scaler.pkl"),             "rb") as f: scaler    = pickle.load(f)
    with open(os.path.join(base, "feature_cols.pkl"),       "rb") as f: feat_cols = pickle.load(f)
    with open(os.path.join(base, "training_reference.pkl"), "rb") as f: ref_df    = pickle.load(f)
    return {"model": model, "scaler": scaler, "feat_cols": feat_cols, "ref_df": ref_df}

def load_all_feature_cols() -> dict:
    cols = {}
    for eq in ["ahu", "fcu", "boiler"]:
        p = os.path.join(BASE_MODELS, eq, "feature_cols.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                cols[eq] = set(pickle.load(f))
    return cols

# ── Step 1: Skip filter ───────────────────────────────────────────────────────

def should_skip(filename: str) -> tuple[bool, str]:
    fname_lower = Path(filename).name.lower()
    for substr in SKIP_FILENAME_SUBSTRINGS:
        if substr in fname_lower:
            return True, f"Skipped — filename contains '{substr}' (output file, not sensor data)"
    return False, ""

# ── Step 2: Equipment detection ───────────────────────────────────────────────

def detect_equipment(path: str, registry_cols: dict) -> tuple[str, str, float]:
    """Returns (equipment, method, confidence)."""
    stem = Path(path).stem.lower()
    for equip, patterns in FILENAME_SIGNATURES.items():
        if any(p in stem for p in patterns):
            return equip, "filename", 1.0

    # Column fallback
    try:
        df = pd.read_csv(path, nrows=500, low_memory=False)
        df.columns = df.columns.str.strip()
        file_cols = {c.lower() for c in df.columns if c.lower() not in EXCLUDE_COLS}

        all_sigs = dict(COLUMN_SIGNATURES)
        for eq, cols in registry_cols.items():
            all_sigs[eq] = cols

        best_equip, best_score = "unknown", 0.0
        for equip, sig_cols in all_sigs.items():
            sig_lower = {c.lower() for c in sig_cols}
            if not sig_lower:
                continue
            score = len(file_cols & sig_lower) / len(sig_lower)
            if score > best_score:
                best_score, best_equip = score, equip

        return best_equip, "column_signature", round(best_score, 3)
    except Exception:
        return "unknown", "error", 0.0

# ── Step 3: PSI gate (raw values, before scaling) ─────────────────────────────

def psi_single(ref_series: pd.Series, new_series: pd.Series) -> float:
    # Force numeric — reference pkl may contain residual string/object values
    ref_num = pd.to_numeric(ref_series, errors="coerce").dropna()
    new_num = pd.to_numeric(new_series, errors="coerce").dropna()
    if len(ref_num) == 0 or len(new_num) == 0:
        return 0.0
    quantiles  = np.linspace(0, 100, N_BINS + 1)
    bin_edges  = np.unique(np.percentile(ref_num, quantiles))
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf
    ref_c = np.histogram(ref_num, bins=bin_edges)[0]
    new_c = np.histogram(new_num, bins=bin_edges)[0]
    eps   = 1e-6
    ref_p = np.clip(ref_c / ref_c.sum(), eps, None)
    new_p = np.clip(new_c / new_c.sum(), eps, None)
    return float(np.sum((new_p - ref_p) * np.log(new_p / ref_p)))

def psi_gate(ref_df: pd.DataFrame, new_df: pd.DataFrame, feat_cols: list) -> dict:
    shared = [c for c in feat_cols if c in ref_df.columns and c in new_df.columns]
    per_feature = {c: psi_single(ref_df[c], new_df[c]) for c in shared}
    overall = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    if overall < PSI_GREEN:
        gate, advice = "GREEN", "Deploy immediately."
    elif overall < PSI_AMBER:
        gate, advice = "AMBER", "Collect adaptation data before deployment."
    else:
        gate, advice = "RED",   "Extreme drift — adaptation required."
    return {
        "overall_psi":     overall,
        "gate":            gate,
        "advice":          advice,
        "per_feature_psi": per_feature,
        "n_red":           sum(1 for v in per_feature.values() if v >= PSI_AMBER),
        "n_amber":         sum(1 for v in per_feature.values() if PSI_GREEN <= v < PSI_AMBER),
    }

# ── Step 4: Preprocessing ─────────────────────────────────────────────────────

def safe_load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=EVAL_CAP, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    return df

def preprocess(df: pd.DataFrame, equipment: str) -> pd.DataFrame:
    if equipment == "fcu":
        df = df.rename(columns=FCU_COL_MAP)
    if equipment == "boiler":
        for col in BOILER_STRUCTURAL_NAN_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        df = df.rename(columns=BOILER_COL_MAP)
    if equipment == "ahu":
        # SD-AHU files use different column names — rename to AHU model feature names
        df = df.rename(columns=SDAHU_COL_MAP)
    # Force all columns to numeric (catches residual Excel errors / str values)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ── Full pipeline for one file ─────────────────────────────────────────────────

def run_pipeline(path: str, equipment_override: str | None = None,
                 registry_cols: dict | None = None) -> dict:
    fname = os.path.basename(path)
    result = {
        "file":      fname,
        "skipped":   False,
        "equipment": None,
        "psi":       None,
        "gate":      None,
        "predictions": None,
        "error":     None,
    }

    # ── Step 1: skip filter ──────────────────────────────────────────────────
    skip, reason = should_skip(fname)
    if skip:
        result["skipped"] = True
        result["error"]   = reason
        return result

    if registry_cols is None:
        registry_cols = load_all_feature_cols()

    # ── Step 2: detect equipment ─────────────────────────────────────────────
    if equipment_override:
        equipment, det_method, det_conf = equipment_override, "override", 1.0
    else:
        equipment, det_method, det_conf = detect_equipment(path, registry_cols)

    result["equipment"]      = equipment
    result["det_method"]     = det_method
    result["det_confidence"] = det_conf

    # Check model exists
    model_path = os.path.join(BASE_MODELS, equipment, "rf_model.pkl")
    if not os.path.exists(model_path):
        result["error"] = (
            f"CAT3 — No registry model for '{equipment}'. "
            f"Architectural mismatch or unsupported equipment type."
        )
        return result

    # ── Load registry ────────────────────────────────────────────────────────
    try:
        reg = load_registry(equipment)
    except FileNotFoundError as e:
        result["error"] = str(e)
        return result

    feat_cols = reg["feat_cols"]

    # ── Load and preprocess new file ─────────────────────────────────────────
    df_raw = safe_load(path)
    df_raw = preprocess(df_raw, equipment)

    available = [c for c in feat_cols if c in df_raw.columns]
    missing   = [c for c in feat_cols if c not in df_raw.columns]
    df_feat   = df_raw[available].dropna()

    if len(available) == 0:
        result["error"] = (
            f"No matching feature columns found. "
            f"Model expects: {feat_cols}. "
            f"File has: {list(df_raw.columns[:10])}..."
        )
        return result

    if len(df_feat) == 0:
        result["error"] = "No usable rows after preprocessing."
        return result

    # ── Step 3: PSI gate on RAW values ───────────────────────────────────────
    psi_result = psi_gate(reg["ref_df"], df_feat, available)
    result["psi"]  = psi_result["overall_psi"]
    result["gate"] = psi_result["gate"]
    result["psi_detail"] = psi_result

    # ── Step 4 & 5: Scale then predict ───────────────────────────────────────
    if missing:
        for col in missing:
            if col in reg["ref_df"].columns:
                fill_val = pd.to_numeric(reg["ref_df"][col], errors="coerce").mean()
            else:
                fill_val = 0.0
            df_feat = df_feat.copy()
            df_feat[col] = fill_val
    df_feat = df_feat[feat_cols]
    X_raw = df_feat.values.astype(float)
    X_scaled = reg["scaler"].transform(X_raw)

    proba    = reg["model"].predict_proba(X_scaled)
    preds    = reg["model"].predict(X_scaled)
    classes  = reg["model"].classes_

    # Aggregate over all rows
    mean_proba   = proba.mean(axis=0)
    pred_class   = int(classes[np.argmax(mean_proba)])
    pred_label   = FAULT_LABELS.get(pred_class, f"Class {pred_class}")
    confidence   = float(mean_proba.max())

    # Per-class breakdown
    class_breakdown = {
        int(cls): {
            "label": FAULT_LABELS.get(int(cls), f"Class {cls}"),
            "mean_probability": round(float(p), 4),
        }
        for cls, p in zip(classes, mean_proba)
    }

    # Row-level majority vote (for context)
    from collections import Counter
    vote_counts  = Counter(int(p) for p in preds)
    dominant_cls = vote_counts.most_common(1)[0][0]

    result["predictions"] = {
        "rows_evaluated":   len(df_feat),
        "features_used":    len(available),
        "features_missing": missing,
        "predicted_class":  pred_class,
        "predicted_label":  pred_label,
        "confidence":       round(confidence, 4),
        "class_breakdown":  class_breakdown,
        "row_vote_dominant_class": dominant_cls,
        "row_vote_counts":  dict(vote_counts),
    }

    return result

# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(r: dict):
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"FILE: {r['file']}")

    if r["skipped"]:
        print(f"  ⏭  SKIPPED: {r['error']}")
        return

    if r["error"] and not r["predictions"]:
        print(f"  ✗  ERROR: {r['error']}")
        return

    eq = (r.get("equipment") or "unknown").upper()
    print(f"  Equipment  : {eq}  (detected via {r.get('det_method','?')}, "
          f"confidence {r.get('det_confidence',0):.0%})")

    # PSI
    psi  = r.get("psi", 0)
    gate = r.get("gate", "?")
    icons = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
    icon  = icons.get(gate, "❓")
    print(f"  PSI Gate   : {icon} {gate}  (overall PSI = {psi:.4f})")

    # Per-feature PSI if any are elevated
    detail = r.get("psi_detail", {})
    pf = detail.get("per_feature_psi", {})
    elevated = {k: v for k, v in pf.items() if v >= PSI_GREEN}
    if elevated:
        print(f"  Elevated PSI features ({len(elevated)}):")
        for col, val in sorted(elevated.items(), key=lambda x: -x[1])[:5]:
            short = col[:50]
            print(f"    {short:<50}  {val:.3f}  {icons.get('RED' if val>=PSI_AMBER else 'AMBER','')}")

    # Predictions
    pred = r.get("predictions", {})
    if pred:
        print(f"\n  ── PREDICTION ──────────────────────────────────────")
        print(f"  Rows evaluated : {pred['rows_evaluated']:,}")
        print(f"  Features used  : {pred['features_used']}")
        if pred["features_missing"]:
            print(f"  Features missing: {pred['features_missing']}")
        print(f"\n  Predicted class: {pred['predicted_class']} — {pred['predicted_label']}")
        print(f"  Confidence     : {pred['confidence']:.1%}")
        print(f"\n  Class probabilities:")
        for cls_int, info in sorted(pred["class_breakdown"].items()):
            bar_len = int(info["mean_probability"] * 30)
            bar     = "█" * bar_len
            print(f"    Class {cls_int} ({info['label']:<38})  "
                  f"{info['mean_probability']:>6.1%}  {bar}")

        votes = pred.get("row_vote_counts", {})
        total = sum(votes.values())
        if total:
            print(f"\n  Row-level vote distribution ({total:,} rows):")
            for cls_int in sorted(votes):
                pct = votes[cls_int] / total
                print(f"    Class {cls_int}: {votes[cls_int]:>7,} rows  ({pct:.1%})")

    if r.get("error"):
        print(f"\n  ⚠  {r['error']}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal HVAC FDD Prediction Pipeline"
    )
    parser.add_argument("target", help="CSV file or folder path")
    parser.add_argument("--equipment", "-e", default=None,
                        help="Override equipment detection (ahu|fcu|boiler)")
    args = parser.parse_args()

    registry_cols = load_all_feature_cols()
    print(f"Registry models loaded: {list(registry_cols.keys())}")

    target = args.target
    if os.path.isfile(target) and target.endswith(".csv"):
        r = run_pipeline(target, args.equipment, registry_cols)
        print_result(r)

    elif os.path.isdir(target):
        csvs = sorted(glob.glob(os.path.join(target, "*.csv")))
        print(f"\nBatch mode: {len(csvs)} CSV files in {target}\n")
        summary = {"GREEN": [], "AMBER": [], "RED": [], "SKIPPED": [], "ERROR": []}

        for path in csvs:
            r = run_pipeline(path, args.equipment, registry_cols)
            print_result(r)
            if r["skipped"]:
                summary["SKIPPED"].append(r["file"])
            elif r["error"] and not r["predictions"]:
                summary["ERROR"].append(r["file"])
            else:
                summary[r.get("gate", "ERROR")].append(r["file"])

        print(f"\n{'='*65}")
        print("BATCH SUMMARY")
        for status, files in summary.items():
            if files:
                print(f"  {status:8}: {len(files)} file(s)")

    else:
        print(f"ERROR: '{target}' is not a CSV file or directory.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
