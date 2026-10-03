"""
Step 3 — Equipment Detector
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Identifies equipment type (AHU / FCU / Boiler / RTU) from:
  1. Filename pattern matching (fast, reliable when filenames follow conventions)
  2. Column signature matching (fallback — checks which model's feature set fits best)

Usage:
  python pipeline/equipment_detector.py path/to/file.csv
  python pipeline/equipment_detector.py path/to/folder/      # detects all CSVs in folder
"""

import os
import sys
import glob
import pickle
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODELS = os.path.join(REPO_ROOT, 'models', 'registry')
EVAL_CAP    = 15_000
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
EXCLUDE_COLS = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
    "actual_class", "actual_label", "predicted_class", "predicted_label",
}

# ── Filename signature patterns ────────────────────────────────────────────────
# Each entry: list of substrings (any match → equipment type)
# Checked case-insensitively against the filename stem.

FILENAME_SIGNATURES = {
    "boiler": ["boiler", "boilerplant", "hot_water"],
    "fcu":    ["fcu_", "fcu-"],
    "rtu":    ["rtu"],
    "ahu":    ["mzvav", "szcav", "szvav", "ahu"],
}

# ── Column signature matching ──────────────────────────────────────────────────
# For each equipment type, the canonical feature columns used by its registry model.
# Loaded dynamically from saved feature_cols.pkl; these are fallback hardcoded defaults.

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
        # FCU native column names (before rename)
        "FCU_CVLV", "FCU_HVLV", "FCU_MAT", "FCU_DMPR",
        "FCU_OAT", "FCU_RAT", "FCU_SPD", "FCU_DAT",
        # also accept AHU-equivalent names (after rename)
        "AHU: Cooling Coil Valve Control Signal",
        "AHU: Heating Coil Valve Control Signal",
        "AHU: Mixed Air Temperature",
        "AHU: Outdoor Air Damper Control Signal",
        "AHU: Outdoor Air Temperature",
        "AHU: Return Air Temperature",
        "AHU: Supply Air Fan Speed Control Signal",
        "AHU: Supply Air Temperature",
    },
    "boiler": {
        "OA_TEMP", "HWL_SW_TEMP", "HWL_RW_TEMP", "HWL_DP",
        "BOI_SW_TEMP_1", "BOI_RW_TEMP_1", "PM_SPD_1",
        "BOI_FLOW_1", "BOI_GAS_CSUM_1",
    },
    "rtu": {
        "RTU: Circuit 1 Discharge Pressure",
        "RTU: Circuit 1 Suction Pressure",
        "RTU: Supply Air Temperature",
        "RTU: Return Air Temperature",
        "RTU: Compressor 1 On/Off Status",
    },
    "sdahu": {
        "CHWC_VLV", "SF_SPD", "RF_SPD", "MA_TEMP",
        "OA_DMPR", "SA_TEMP", "RA_TEMP", "OA_TEMP",
    },
}

# ── Load saved feature cols from registry (override hardcoded defaults) ────────

def load_registry_feature_cols() -> dict:
    """Load feature_cols.pkl from each registry subfolder."""
    registry_cols = {}
    for equip in ["ahu", "fcu", "boiler"]:
        pkl_path = os.path.join(BASE_MODELS, equip, "feature_cols.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                registry_cols[equip] = set(pickle.load(f))
    return registry_cols

# ── Detection logic ────────────────────────────────────────────────────────────

def detect_by_filename(filename: str) -> str | None:
    """Return equipment type string or None if no match."""
    stem = Path(filename).stem.lower()
    for equip, patterns in FILENAME_SIGNATURES.items():
        if any(p in stem for p in patterns):
            return equip
    return None


def detect_by_columns(df_cols: list, registry_cols: dict) -> tuple[str, float]:
    """
    Score each equipment type by overlap fraction of its required columns
    vs the file's actual columns.
    Returns (best_equipment, overlap_fraction).
    """
    file_cols_lower = {c.lower() for c in df_cols}

    # Merge registry cols with hardcoded signatures
    all_sigs = dict(COLUMN_SIGNATURES)
    for equip, cols in registry_cols.items():
        all_sigs[equip] = cols

    best_equip = "unknown"
    best_score = 0.0

    for equip, sig_cols in all_sigs.items():
        sig_lower = {c.lower() for c in sig_cols}
        if not sig_lower:
            continue
        overlap = len(file_cols_lower & sig_lower) / len(sig_lower)
        if overlap > best_score:
            best_score = overlap
            best_equip = equip

    return best_equip, best_score


def detect_equipment(path: str, registry_cols: dict | None = None) -> dict:
    """
    Full detection pipeline for one CSV file.
    Returns a result dict with: file, method, equipment, confidence, notes.
    """
    if registry_cols is None:
        registry_cols = load_registry_feature_cols()

    fname = os.path.basename(path)
    result = {
        "file": fname,
        "path": path,
        "equipment": "unknown",
        "method": "none",
        "confidence": 0.0,
        "notes": [],
        "model_available": False,
    }

    # Step 1: filename pattern
    equip_fn = detect_by_filename(fname)
    if equip_fn:
        result["equipment"] = equip_fn
        result["method"]    = "filename"
        result["confidence"] = 1.0
        result["notes"].append(f"Filename matched pattern for '{equip_fn}'")
    else:
        result["notes"].append("No filename pattern match — falling back to column signature")

        # Step 2: load file and check columns
        try:
            df = pd.read_csv(path, nrows=EVAL_CAP, low_memory=False)
            df.columns = df.columns.str.strip()
            df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
            feat_cols = [c for c in df.columns if c.lower() not in EXCLUDE_COLS]

            equip_col, score = detect_by_columns(feat_cols, registry_cols)
            result["equipment"]  = equip_col
            result["method"]     = "column_signature"
            result["confidence"] = round(score, 3)
            result["notes"].append(
                f"Best column match: '{equip_col}' (overlap {score:.1%})"
            )
        except Exception as e:
            result["notes"].append(f"Error reading file: {e}")
            return result

    # Check if a registry model exists for this equipment type
    model_path = os.path.join(BASE_MODELS, result["equipment"], "rf_model.pkl")
    result["model_available"] = os.path.exists(model_path)
    if not result["model_available"]:
        result["notes"].append(
            f"No registry model found for '{result['equipment']}' "
            f"— will be flagged as CAT3 (architectural mismatch)"
        )

    return result


def format_result(r: dict) -> str:
    status = "✓ MODEL READY" if r["model_available"] else "✗ NO MODEL (CAT3)"
    lines = [
        f"  File      : {r['file']}",
        f"  Equipment : {r['equipment'].upper():10}  [{status}]",
        f"  Method    : {r['method']}  (confidence: {r['confidence']:.0%})",
        f"  Notes     : {'; '.join(r['notes'])}",
    ]
    return "\n".join(lines)

# ── Batch detection ────────────────────────────────────────────────────────────

def detect_folder(folder: str):
    registry_cols = load_registry_feature_cols()
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csvs:
        print(f"No CSV files found in: {folder}")
        return

    print(f"\nDetecting equipment for {len(csvs)} files in: {folder}\n")
    summary = {"ahu": [], "fcu": [], "boiler": [], "rtu": [], "sdahu": [], "unknown": []}

    for path in csvs:
        r = detect_equipment(path, registry_cols)
        print(format_result(r))
        print()
        summary[r["equipment"]].append(r["file"])

    print("── SUMMARY ──────────────────────────────────────────")
    for equip, files in summary.items():
        if files:
            model_path = os.path.join(BASE_MODELS, equip, "rf_model.pkl")
            has_model  = "✓" if os.path.exists(model_path) else "✗"
            print(f"  {equip.upper():8} {has_model}  {len(files)} file(s)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    registry_cols = load_registry_feature_cols()
    print(f"Registry models loaded: {list(registry_cols.keys())}\n")

    if not args:
        print("Usage:")
        print("  python pipeline/equipment_detector.py path/to/file.csv")
        print("  python pipeline/equipment_detector.py path/to/folder/")
        return

    for target in args:
        if os.path.isfile(target) and target.endswith(".csv"):
            r = detect_equipment(target, registry_cols)
            print("="*60)
            print(format_result(r))
        elif os.path.isdir(target):
            detect_folder(target)
        else:
            print(f"Skipping: '{target}' is not a CSV file or directory")

    print("\n" + "="*60)
    print("Detection complete.")


if __name__ == "__main__":
    main()
