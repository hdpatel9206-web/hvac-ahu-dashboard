"""
Step 2 — Specialist Model Trainer
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Trains one Random Forest model per equipment type using verified fault mappings.
Saves each model to:  models/registry/<equipment>/rf_model.pkl
                      models/registry/<equipment>/scaler.pkl
                      models/registry/<equipment>/feature_cols.pkl
                      models/registry/<equipment>/training_reference.pkl  (PSI baseline)
                      models/registry/<equipment>/training_report.json

Universal fault classes (all equipment):
  0 = Healthy
  1 = Sensor Bias
  2 = Valve / Coil / Heat Exchanger Fault
  3 = Control Fault

Usage:
  python pipeline/specialist_trainer.py            # train all equipment types
  python pipeline/specialist_trainer.py ahu        # train AHU only
  python pipeline/specialist_trainer.py fcu        # train FCU only
  python pipeline/specialist_trainer.py boiler     # train Boiler only
"""

import os, sys, json, glob, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA   = os.path.join(REPO_ROOT, 'data')
BASE_MODELS = os.path.join(REPO_ROOT, 'models', 'registry')

FOLDERS = {
    "ahu":    BASE_DATA,                          # ASHRAE root: MZVAV-1, MZVAV-2-1, SZCAV …
    "fcu":    os.path.join(BASE_DATA, "fcu"),
    "boiler": os.path.join(BASE_DATA, "boiler"),
}

EVAL_CAP      = 15_000     # max rows per file during eval / reference sampling
ADAPT_CAP     = 50_000     # max rows per file for training (memory safe)
EXCEL_ERRORS  = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

EXCLUDE_COLS  = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
}

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

# ── AHU configuration ─────────────────────────────────────────────────────────

# Only ASHRAE files (no sdahu) for the primary AHU model
AHU_FILES = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv", "MZVAV-2-2.csv", "SZVAV.csv"]

# Verified 11-column intersection (confirmed by Step 1)
AHU_SHARED_COLS = [
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
]

# ASHRAE ground-truth → universal class
AHU_FAULT_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# ── FCU configuration ─────────────────────────────────────────────────────────

# Column rename: FCU native name → AHU equivalent name (our model feature names)
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

# Fault label substring → universal class (applied to 'label' column if present,
# else inferred from filename)
FCU_FAULT_MAP_SUBSTR = {
    "faultfree":              0,
    "sensorbias":             1,
    "fouling":                2,
    "vlvleak":                2,
    "vlvstuck":               2,
    "filterrestriction":      2,
    "fanoutletblockage":      2,
    "oablockage":             2,
    "oadmprleak":             2,
    "oadmprstuck":            2,
    "control_coolingreverse": 3,
    "heatingreverse":         3,
    "unstable":               3,
}

# ── Boiler configuration ──────────────────────────────────────────────────────

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

# Fault label substring → universal class (matched against filename lowercase)
# ORDER MATTERS: specific substrings before the generic "boilerplant" catch-all.
# Every file starts with "boilerplant_" so it must be last.
BOILER_FAULT_MAP_SUBSTR = {
    "boiler_bias":             1,
    "hot_water_temp_bias":     1,
    "hot_water_pressure_bias": 1,
    "boiler_foul":             2,
    "boiler_pi":               3,
    "boilerplant":             0,   # catch-all — healthy baseline file only
}

# Boiler files 5-8 and 13-16 have structural NaN in PM_POW_1/2 — fill with 0
BOILER_STRUCTURAL_NAN_COLS = ["PM_POW_1", "PM_POW_2"]

# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_load(path, nrows=ADAPT_CAP):
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    return df


def save_artifacts(equipment, model, scaler, feature_cols, reference_df, report):
    out_dir = os.path.join(BASE_MODELS, equipment)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    # Save reference sample for PSI computation (Step 4)
    ref_sample = reference_df[feature_cols].dropna().sample(
        min(5000, len(reference_df)), random_state=42
    )
    with open(os.path.join(out_dir, "training_reference.pkl"), "wb") as f:
        pickle.dump(ref_sample, f)
    with open(os.path.join(out_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Saved to {out_dir}")


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    f1_macro = f1_score(y, preds, average="macro", zero_division=0)
    report = classification_report(y, preds, zero_division=0)
    print(f"  {label} macro F1 = {f1_macro:.4f}")
    print(report)
    return float(f1_macro), report

# ── AHU Trainer ───────────────────────────────────────────────────────────────

def train_ahu():
    print("\n" + "="*70)
    print("TRAINING: AHU Specialist Model")
    print("="*70)

    frames = []
    for fname in AHU_FILES:
        path = os.path.join(FOLDERS["ahu"], fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        df = safe_load(path)
        df.columns = df.columns.str.strip()

        # Map fault label
        label_col = "Fault Detection Ground Truth"
        if label_col not in df.columns:
            print(f"  [SKIP] {fname} — no label column")
            continue
        df["universal_label"] = df[label_col].map(AHU_FAULT_MAP)
        df = df.dropna(subset=["universal_label"])
        df["universal_label"] = df["universal_label"].astype(int)

        # Keep only shared columns
        available = [c for c in AHU_SHARED_COLS if c in df.columns]
        missing   = [c for c in AHU_SHARED_COLS if c not in df.columns]
        if missing:
            print(f"  [WARN] {fname} missing: {missing}")
        df = df[available + ["universal_label"]].dropna()
        print(f"  Loaded {fname}: {len(df):,} rows, {df['universal_label'].value_counts().to_dict()}")
        frames.append(df)

    if not frames:
        print("  ERROR: No AHU data loaded.")
        return

    data = pd.concat(frames, ignore_index=True)
    # Align to exact shared cols present in all frames
    feat_cols = [c for c in AHU_SHARED_COLS if c in data.columns]
    X = data[feat_cols].values
    y = data["universal_label"].values

    print(f"\n  Total training rows: {len(X):,}")
    print(f"  Feature columns   : {len(feat_cols)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_sc, y)

    f1_train, rpt = evaluate(model, X_sc, y, label="In-sample")

    # 5-fold CV
    cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"  5-fold CV macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    report = {
        "equipment": "ahu",
        "files_used": [f for f in AHU_FILES if os.path.exists(os.path.join(FOLDERS["ahu"], f))],
        "feature_cols": feat_cols,
        "training_rows": int(len(X)),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "f1_insample": f1_train,
        "f1_cv_mean": float(cv_scores.mean()),
        "f1_cv_std": float(cv_scores.std()),
        "classification_report": rpt,
    }
    save_artifacts("ahu", model, scaler, feat_cols, data, report)

# ── FCU Trainer ───────────────────────────────────────────────────────────────

def infer_fcu_label(filename: str) -> int:
    fname_lower = filename.lower()
    for substr, cls in FCU_FAULT_MAP_SUBSTR.items():
        if substr in fname_lower:
            return cls
    return -1   # unknown


def train_fcu():
    print("\n" + "="*70)
    print("TRAINING: FCU Specialist Model")
    print("="*70)

    csv_files = sorted(glob.glob(os.path.join(FOLDERS["fcu"], "*.csv")))
    print(f"  Found {len(csv_files)} FCU files")

    frames = []
    for path in csv_files:
        fname = os.path.basename(path)
        cls = infer_fcu_label(fname)
        if cls == -1:
            print(f"  [SKIP] {fname} — cannot infer fault class")
            continue

        df = safe_load(path, nrows=ADAPT_CAP)
        df.columns = df.columns.str.strip()

        # Rename FCU columns → AHU equivalent names
        df = df.rename(columns=FCU_COL_MAP)

        feat_cols = [c for c in FCU_COL_MAP.values() if c in df.columns]
        missing   = [c for c in FCU_COL_MAP.values() if c not in df.columns]
        if missing:
            print(f"  [WARN] {fname} missing after rename: {missing}")

        df = df[feat_cols].copy()
        df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
        df = df.dropna()
        df["universal_label"] = cls

        print(f"  {fname}: class={cls}, {len(df):,} rows")
        frames.append(df)

    if not frames:
        print("  ERROR: No FCU data loaded.")
        return

    data = pd.concat(frames, ignore_index=True)

    # Cap total training size to avoid memory issues (25M rows across all files)
    if len(data) > ADAPT_CAP * 10:
        sampled_parts = []
        for cls_val, grp in data.groupby("universal_label"):
            sampled_parts.append(grp.sample(min(len(grp), 50_000), random_state=42))
        data = pd.concat(sampled_parts, ignore_index=True)
        print(f"  [CAP] Down-sampled to {len(data):,} rows (balanced per class)")

    feat_cols = [c for c in FCU_COL_MAP.values() if c in data.columns]
    X = data[feat_cols].values
    y = data["universal_label"].values

    print(f"\n  Total training rows: {len(X):,}")
    print(f"  Feature columns   : {len(feat_cols)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_sc, y)

    f1_train, rpt = evaluate(model, X_sc, y, label="In-sample")

    cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"  5-fold CV macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    report = {
        "equipment": "fcu",
        "files_used": [os.path.basename(p) for p in csv_files],
        "feature_cols": feat_cols,
        "training_rows": int(len(X)),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "f1_insample": f1_train,
        "f1_cv_mean": float(cv_scores.mean()),
        "f1_cv_std": float(cv_scores.std()),
        "classification_report": rpt,
    }
    save_artifacts("fcu", model, scaler, feat_cols, data, report)

# ── Boiler Trainer ────────────────────────────────────────────────────────────

def infer_boiler_label(filename: str) -> int:
    fname_lower = filename.lower()
    for substr, cls in BOILER_FAULT_MAP_SUBSTR.items():
        if substr in fname_lower:
            return cls
    return -1


def train_boiler():
    print("\n" + "="*70)
    print("TRAINING: Boiler Specialist Model")
    print("="*70)

    csv_files = sorted(glob.glob(os.path.join(FOLDERS["boiler"], "*.csv")))
    print(f"  Found {len(csv_files)} Boiler files")

    frames = []
    for path in csv_files:
        fname = os.path.basename(path)
        cls = infer_boiler_label(fname)
        if cls == -1:
            print(f"  [SKIP] {fname} — cannot infer fault class")
            continue

        df = safe_load(path, nrows=ADAPT_CAP)
        df.columns = df.columns.str.strip()

        # Fill structural NaN columns before any processing
        for col in BOILER_STRUCTURAL_NAN_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Rename boiler columns → canonical names
        df = df.rename(columns=BOILER_COL_MAP)

        feat_cols = [c for c in BOILER_COL_MAP.values() if c in df.columns]
        missing   = [c for c in BOILER_COL_MAP.values() if c not in df.columns]
        if missing:
            print(f"  [WARN] {fname} missing after rename: {missing}")

        df = df[feat_cols].copy()
        df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
        df = df.dropna()
        df["universal_label"] = cls

        print(f"  {fname}: class={cls}, {len(df):,} rows")
        frames.append(df)

    if not frames:
        print("  ERROR: No Boiler data loaded.")
        return

    data = pd.concat(frames, ignore_index=True)

    feat_cols = [c for c in BOILER_COL_MAP.values() if c in data.columns]
    X = data[feat_cols].values
    y = data["universal_label"].values

    print(f"\n  Total training rows: {len(X):,}")
    print(f"  Feature columns   : {len(feat_cols)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_sc, y)

    f1_train, rpt = evaluate(model, X_sc, y, label="In-sample")

    cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"  5-fold CV macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    report = {
        "equipment": "boiler",
        "files_used": [os.path.basename(p) for p in csv_files],
        "feature_cols": feat_cols,
        "training_rows": int(len(X)),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "f1_insample": f1_train,
        "f1_cv_mean": float(cv_scores.mean()),
        "f1_cv_std": float(cv_scores.std()),
        "classification_report": rpt,
    }
    save_artifacts("boiler", model, scaler, feat_cols, data, report)

# ── Main ──────────────────────────────────────────────────────────────────────

TRAINERS = {
    "ahu":    train_ahu,
    "fcu":    train_fcu,
    "boiler": train_boiler,
}

def main():
    targets = sys.argv[1:] if sys.argv[1:] else list(TRAINERS.keys())
    for t in targets:
        t = t.lower()
        if t in TRAINERS:
            TRAINERS[t]()
        else:
            print(f"Unknown equipment: {t}. Choose from: {list(TRAINERS.keys())}")

    print("\n" + "="*70)
    print("Training complete. Registry saved to:")
    print(f"  {BASE_MODELS}")
    print("="*70)

if __name__ == "__main__":
    main()
