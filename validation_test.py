"""
Step 6 — Validation Test
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Runs the complete pipeline on one representative file per equipment type,
confirms correct routing, PSI gate, and prediction, then computes F1
against ground-truth labels where available.

Test matrix:
  AHU    — MZVAV-1.csv          (ASHRAE ground truth available)
  FCU    — FCU_FaultFree.csv    (filename encodes class 0)
  FCU    — FCU_SensorBias_RMTemp_+2C.csv  (filename encodes class 1)
  Boiler — BoilerPlant.csv      (filename encodes class 0)
  Boiler — BoilerPlant_boiler_bias_2.csv  (filename encodes class 1)
  SD-AHU — AHU_annual.csv       (healthy baseline, class 0)
  RTU    — RTU.csv              (CAT3 — no model expected)

Usage:
  python validation_test.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DATA   = r"C:\Users\hrslp\Desktop\thesis\data"
BASE_MODELS = r"C:\Users\hrslp\Desktop\thesis\models\registry"

# ── Import pipeline functions ──────────────────────────────────────────────────
# Add thesis folder to path so we can import predict_pipeline directly
sys.path.insert(0, r"C:\Users\hrslp\Desktop\thesis")
from predict_pipeline import (
    run_pipeline, load_all_feature_cols,
    safe_load, preprocess,
    BOILER_COL_MAP, SDAHU_COL_MAP, FCU_COL_MAP,
    BOILER_STRUCTURAL_NAN_COLS,
)

# ── AHU fault map (ASHRAE ground truth → universal class) ─────────────────────
AHU_FAULT_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# ── FCU label inference from filename ─────────────────────────────────────────
FCU_FAULT_MAP_SUBSTR = {
    "faultfree": 0, "sensorbias": 1,
    "fouling": 2, "vlvleak": 2, "vlvstuck": 2,
    "filterrestriction": 2, "fanoutletblockage": 2,
    "oablockage": 2, "oadmprleak": 2, "oadmprstuck": 2,
    "control_coolingreverse": 3, "heatingreverse": 3, "unstable": 3,
}

# ── Boiler label inference from filename ──────────────────────────────────────
BOILER_FAULT_MAP_SUBSTR = {
    "boiler_bias": 1, "hot_water_temp_bias": 1, "hot_water_pressure_bias": 1,
    "boiler_foul": 2, "boiler_pi": 3, "boilerplant": 0,
}

EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
EVAL_CAP     = 15_000
EXCLUDE_COLS = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
}

FAULT_LABELS = {
    0: "Healthy", 1: "Sensor Bias",
    2: "Valve/Coil/HX Fault", 3: "Control Fault",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_label_from_filename(fname: str, fault_map: dict) -> int:
    fl = fname.lower()
    for substr, cls in fault_map.items():
        if substr in fl:
            return cls
    return -1


def get_ahu_labels(path: str) -> np.ndarray | None:
    df = pd.read_csv(path, nrows=EVAL_CAP, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    label_col = "Fault Detection Ground Truth"
    if label_col not in df.columns:
        return None
    labels = df[label_col].map(AHU_FAULT_MAP).dropna().astype(int).values
    return labels[:EVAL_CAP]


def get_fcu_labels(path: str, true_class: int, n_rows: int) -> np.ndarray:
    return np.full(n_rows, true_class, dtype=int)


def get_boiler_labels(path: str, true_class: int, n_rows: int) -> np.ndarray:
    return np.full(n_rows, true_class, dtype=int)


def row_predictions(path: str, equipment: str) -> np.ndarray | None:
    """Re-run prediction and return per-row predicted classes."""
    reg_path = os.path.join(BASE_MODELS, equipment)
    try:
        with open(os.path.join(reg_path, "rf_model.pkl"),          "rb") as f: model     = pickle.load(f)
        with open(os.path.join(reg_path, "scaler.pkl"),             "rb") as f: scaler    = pickle.load(f)
        with open(os.path.join(reg_path, "feature_cols.pkl"),       "rb") as f: feat_cols = pickle.load(f)
        with open(os.path.join(reg_path, "training_reference.pkl"), "rb") as f: ref_df    = pickle.load(f)
    except FileNotFoundError:
        return None

    df = safe_load(path)
    df = preprocess(df, equipment)

    available = [c for c in feat_cols if c in df.columns]
    missing   = [c for c in feat_cols if c not in df.columns]
    df_feat   = df[available].dropna()
    if len(df_feat) == 0:
        return None

    if missing:
        for col in missing:
            fill_val = pd.to_numeric(ref_df[col], errors="coerce").mean() if col in ref_df.columns else 0.0
            df_feat = df_feat.copy()
            df_feat[col] = fill_val

    df_feat  = df_feat[feat_cols]
    X_scaled = scaler.transform(df_feat.values.astype(float))
    return model.predict(X_scaled)

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label":     "AHU — MZVAV-1 (mixed faults)",
        "path":      os.path.join(BASE_DATA, "MZVAV-1.csv"),
        "equipment": "ahu",
        "cat":       "CAT1",
        "label_src": "ashrae_groundtruth",
    },
    {
        "label":     "AHU — SZCAV (mixed faults)",
        "path":      os.path.join(BASE_DATA, "SZCAV.csv"),
        "equipment": "ahu",
        "cat":       "CAT1",
        "label_src": "ashrae_groundtruth",
    },
    {
        "label":     "FCU — FaultFree (class 0)",
        "path":      os.path.join(BASE_DATA, "fcu", "FCU_FaultFree.csv"),
        "equipment": "fcu",
        "cat":       "CAT2",
        "label_src": "filename",
        "true_class": 0,
        "fault_map":  FCU_FAULT_MAP_SUBSTR,
    },
    {
        "label":     "FCU — SensorBias +2C (class 1)",
        "path":      os.path.join(BASE_DATA, "fcu", "FCU_SensorBias_RMTemp_+2C.csv"),
        "equipment": "fcu",
        "cat":       "CAT2",
        "label_src": "filename",
        "true_class": 1,
        "fault_map":  FCU_FAULT_MAP_SUBSTR,
    },
    {
        "label":     "Boiler — BoilerPlant (class 0)",
        "path":      os.path.join(BASE_DATA, "boiler", "BoilerPlant.csv"),
        "equipment": "boiler",
        "cat":       "CAT2",
        "label_src": "filename",
        "true_class": 0,
        "fault_map":  BOILER_FAULT_MAP_SUBSTR,
    },
    {
        "label":     "Boiler — boiler_bias_2 (class 1)",
        "path":      os.path.join(BASE_DATA, "boiler", "BoilerPlant_boiler_bias_2.csv"),
        "equipment": "boiler",
        "cat":       "CAT2",
        "label_src": "filename",
        "true_class": 1,
        "fault_map":  BOILER_FAULT_MAP_SUBSTR,
    },
    {
        "label":     "SD-AHU — AHU_annual (class 0, cross-building)",
        "path":      os.path.join(BASE_DATA, "sdahu", "AHU_annual.csv"),
        "equipment": "ahu",
        "cat":       "CAT1",
        "label_src": "filename",
        "true_class": 0,
        "fault_map":  {},
    },
    {
        "label":     "RTU — RTU.csv (CAT3 expected)",
        "path":      os.path.join(BASE_DATA, "RTU.csv"),
        "equipment": "rtu",
        "cat":       "CAT3",
        "label_src": "none",
    },
]

# ── Run validation ────────────────────────────────────────────────────────────

def run_validation():
    registry_cols = load_all_feature_cols()
    results = []

    print("\n" + "="*70)
    print("STEP 6 — VALIDATION TEST")
    print("Universal HVAC FDD Model Registry")
    print("="*70)

    for tc in TEST_CASES:
        path = tc["path"]
        fname = os.path.basename(path)
        equipment = tc["equipment"]
        cat = tc["cat"]

        print(f"\n{'─'*70}")
        print(f"TEST: {tc['label']}")
        print(f"  File      : {fname}")
        print(f"  Equipment : {equipment.upper()}  [{cat}]")

        if not os.path.exists(path):
            print(f"  SKIP — file not found: {path}")
            continue

        # Run pipeline (routing + PSI + prediction)
        r = run_pipeline(path, equipment_override=equipment, registry_cols=registry_cols)

        # Routing check
        routed_to = r.get("equipment", "none")
        routing_ok = (routed_to == equipment) or (cat == "CAT3" and r.get("error", "").startswith("CAT3"))
        print(f"  Routing   : {'✓ CORRECT' if routing_ok else '✗ WRONG'} → {routed_to.upper()}")

        # PSI
        if r.get("psi") is not None:
            print(f"  PSI Gate  : {r['gate']}  (PSI = {r['psi']:.4f})")
        elif cat == "CAT3":
            print(f"  PSI Gate  : N/A (CAT3 — no model)")

        # Prediction + F1
        if cat == "CAT3":
            expected_error = r.get("error", "")
            print(f"  CAT3 flag : {'✓' if 'CAT3' in expected_error else '✗'}  {expected_error}")
            results.append({"test": tc["label"], "routing": routing_ok,
                            "cat3_correct": "CAT3" in expected_error, "f1": None})
            continue

        pred = r.get("predictions")
        if not pred:
            print(f"  ERROR: {r.get('error')}")
            results.append({"test": tc["label"], "routing": routing_ok, "f1": None})
            continue

        print(f"  Prediction: Class {pred['predicted_class']} — {FAULT_LABELS.get(pred['predicted_class'], '?')}  ({pred['confidence']:.1%})")

        # Get row-level predictions and ground truth for F1
        preds_row = row_predictions(path, equipment)

        if tc["label_src"] == "ashrae_groundtruth":
            y_true = get_ahu_labels(path)
            if y_true is not None and preds_row is not None:
                n = min(len(y_true), len(preds_row))
                f1 = f1_score(y_true[:n], preds_row[:n], average="macro", zero_division=0)
                print(f"  F1 (macro): {f1:.4f}")
                print(f"  Classes in ground truth: {sorted(set(y_true[:n]))}")
                print(classification_report(y_true[:n], preds_row[:n],
                                            target_names=[FAULT_LABELS.get(i, str(i)) for i in sorted(set(y_true[:n]))],
                                            zero_division=0))
                results.append({"test": tc["label"], "routing": routing_ok, "f1": f1})
            else:
                print("  F1: could not compute (no ground truth or predictions)")
                results.append({"test": tc["label"], "routing": routing_ok, "f1": None})

        elif tc["label_src"] == "filename":
            true_class = tc.get("true_class", -1)
            if true_class == -1:
                true_class = infer_label_from_filename(fname, tc.get("fault_map", {}))
            if preds_row is not None and true_class >= 0:
                y_true = np.full(len(preds_row), true_class)
                f1 = f1_score(y_true, preds_row, average="macro", zero_division=0)
                accuracy = (preds_row == true_class).mean()
                print(f"  True class: {true_class} — {FAULT_LABELS.get(true_class, '?')}")
                print(f"  F1 (macro): {f1:.4f}   Accuracy: {accuracy:.1%}")
                print(f"  Row votes : {dict(sorted({int(c): int((preds_row==c).sum()) for c in set(preds_row)}.items()))}")
                results.append({"test": tc["label"], "routing": routing_ok, "f1": f1})
            else:
                results.append({"test": tc["label"], "routing": routing_ok, "f1": None})

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print(f"{'Test':<45}  {'Routing':<10}  {'F1':>6}")
    print(f"{'─'*45}  {'─'*10}  {'─'*6}")
    for r in results:
        routing_str = "✓" if r.get("routing") else "✗"
        f1_str = f"{r['f1']:.4f}" if r["f1"] is not None else "N/A"
        cat3_str = " [CAT3✓]" if r.get("cat3_correct") else ""
        print(f"{r['test'][:45]:<45}  {routing_str:<10}  {f1_str:>6}{cat3_str}")

    routing_pass = sum(1 for r in results if r.get("routing"))
    print(f"\n  Routing correct : {routing_pass}/{len(results)}")
    print(f"  Pipeline status : {'ALL PASS' if routing_pass == len(results) else 'REVIEW NEEDED'}")
    print("="*70)


if __name__ == "__main__":
    run_validation()
