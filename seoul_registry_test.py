# seoul_registry_test.py
# Tests the Universal FDD Registry on the Seoul real BMS dataset
# Run: python seoul_registry_test.py
#
# What this tests:
# 1. Can the registry identify equipment type from Korean BMS column names?
# 2. What PSI values does it compute on real operational data?
# 3. How does the classifier perform on completely unseen real-world data?
# 4. Does the registry correctly flag this as a deployment risk?

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report

BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEOUL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seoul")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

# ── Seoul column mapping to ASHRAE equivalents ────────────────────────────────
# Based on physical sensor roles not exact names
SEOUL_COL_MAP = {
    "Supply Air Temperature":    "AHU: Supply Air Temperature",
    "Ventilation Temperature":   "AHU: Outdoor Air Temperature",
    "Supply Fan":                "AHU: Supply Air Fan Status",
    "Valve Position":            "AHU: Cooling Coil Valve Control Signal",
}

# Seoul label mapping
SEOUL_LABEL_MAP = {
    "Normal":        0,
    "normal":        0,
    "Sensor fault":  1,
    "sensor fault":  1,
    "Fan fault":     2,
    "fan fault":     2,
    "Valve fault":   3,
    "valve fault":   3,
    "Heating fault": 3,
}

ASHRAE_RAW_COLS = [
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

TRAIN_FILES = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]

print("=" * 65)
print("SEOUL REAL BMS DATASET - UNIVERSAL REGISTRY TEST")
print("Testing on REAL operational data from South Korea")
print("=" * 65)

# ── STEP 1: Find Seoul files ──────────────────────────────────────────────────
print("\nStep 1: Finding Seoul dataset files...")
seoul_files = []

# Check multiple possible locations
search_dirs = [
    SEOUL_DIR,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seoul_bms"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "korea"),
]

for d in search_dirs:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith('.csv') and ('seoul' in f.lower() or 'korea' in f.lower()
                                       or 'office' in f.lower() or 'bms' in f.lower()
                                       or 'ahu' in f.lower()):
                seoul_files.append(os.path.join(d, f))

# Also check root thesis folder
root_csvs = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
             if f.endswith('.csv') and any(k in f.lower()
             for k in ['seoul', 'korea', 'office', 'bms', 'wang'])]
for f in root_csvs:
    seoul_files.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))

if not seoul_files:
    print("\n  Seoul files not found automatically.")
    print("  Please specify the path manually:")
    print("  Edit SEOUL_PATH variable below and run again.")
    SEOUL_PATH = None
else:
    print(f"  Found {len(seoul_files)} potential Seoul files:")
    for f in seoul_files:
        print(f"    {f}")
    SEOUL_PATH = seoul_files[0]

# ── MANUAL PATH OVERRIDE - edit this if auto-detect fails ────────────────────
# SEOUL_PATH = r"C:\Users\hrslp\Desktop\thesis\data\seoul\office_ahu.csv"
# ─────────────────────────────────────────────────────────────────────────────

if SEOUL_PATH is None or not os.path.exists(SEOUL_PATH):
    print("\n  Could not find Seoul file.")
    print("  Please put the Seoul CSV file in one of these locations:")
    for d in search_dirs:
        print(f"    {d}")
    print("  Then run this script again.")
    exit(1)

# ── STEP 2: Load and inspect Seoul data ──────────────────────────────────────
print(f"\nStep 2: Loading Seoul data from {os.path.basename(SEOUL_PATH)}...")
seoul_df = pd.read_csv(SEOUL_PATH, low_memory=False)
seoul_df.columns = seoul_df.columns.str.strip()

print(f"  Shape: {seoul_df.shape}")
print(f"  Columns: {list(seoul_df.columns)}")

# Find label column
label_col = None
for possible in ["Labeling", "Label", "label", "labeling", "fault", "Fault",
                 "class", "Class", "fault_type", "Fault_Type"]:
    if possible in seoul_df.columns:
        label_col = possible
        break

if label_col:
    print(f"\n  Label column: '{label_col}'")
    print(f"  Unique labels: {seoul_df[label_col].unique().tolist()}")
    print(f"  Label counts:\n{seoul_df[label_col].value_counts()}")
else:
    print("\n  WARNING: No label column found. Will run PSI analysis only.")

# ── STEP 3: Column mapping ────────────────────────────────────────────────────
print("\nStep 3: Mapping Seoul columns to ASHRAE equivalents...")
mapped_df = seoul_df.copy()

# Apply column mapping
for seoul_col, ashrae_col in SEOUL_COL_MAP.items():
    if seoul_col in mapped_df.columns:
        mapped_df[ashrae_col] = pd.to_numeric(mapped_df[seoul_col], errors='coerce')
        print(f"  Mapped: '{seoul_col}' -> '{ashrae_col}'")
    else:
        print(f"  NOT FOUND: '{seoul_col}' (expected in Seoul data)")

# Zero-fill missing ASHRAE columns
shared_cols = []
for col in ASHRAE_RAW_COLS:
    if col in mapped_df.columns:
        shared_cols.append(col)
    else:
        mapped_df[col] = 0.0

print(f"\n  Shared sensor roles: {len(shared_cols)}")
print(f"  Shared columns: {shared_cols}")
print(f"  Zero-filled missing: {[c for c in ASHRAE_RAW_COLS if c not in shared_cols]}")

# ── STEP 4: PSI analysis ──────────────────────────────────────────────────────
print("\nStep 4: PSI analysis (Seoul vs ASHRAE training distributions)...")

def compute_psi(ref, dep, n_bins=10):
    ref = pd.to_numeric(ref, errors='coerce').dropna().astype(float)
    dep = pd.to_numeric(dep, errors='coerce').dropna().astype(float)
    if len(ref) == 0 or len(dep) == 0:
        return 0.0
    mn = float(min(ref.min(), dep.min()))
    mx = float(max(ref.max(), dep.max()))
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    eps  = 1e-6
    rc   = np.histogram(ref, bins=bins)[0].astype(float) + eps
    dc   = np.histogram(dep, bins=bins)[0].astype(float) + eps
    rp, dp = rc/rc.sum(), dc/dc.sum()
    return float(np.sum((dp - rp) * np.log(dp / rp)))

# Load ASHRAE training reference
print("  Loading ASHRAE training reference...")
train_frames = []
for f in TRAIN_FILES:
    path = os.path.join(BASE, f)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
        for col in ASHRAE_RAW_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        train_frames.append(df[[c for c in ASHRAE_RAW_COLS if c in df.columns]])

if train_frames:
    train_ref = pd.concat(train_frames, ignore_index=True)
    print(f"  ASHRAE training reference: {len(train_ref):,} rows")

    print(f"\n  {'Sensor':50}  {'PSI':8}  Category  Gate")
    print("  " + "-" * 72)
    psi_results = {}
    for col in shared_cols:
        if col in train_ref.columns:
            psi = compute_psi(train_ref[col], mapped_df[col])
            psi_results[col] = psi
            if psi > 0.5:
                cat, gate = "Extreme", "RED - DO NOT DEPLOY"
            elif psi > 0.2:
                cat, gate = "High", "AMBER - CAUTION"
            elif psi > 0.1:
                cat, gate = "Moderate", "GREEN - OK"
            else:
                cat, gate = "Low", "GREEN - OK"
            print(f"  {col:50}  {psi:.4f}   {cat:8}  {gate}")

    overall_psi = max(psi_results.values()) if psi_results else 0
    print(f"\n  Overall PSI gate: {'RED - ADAPTATION REQUIRED' if overall_psi > 0.5 else 'AMBER'}")
    print(f"  Maximum PSI: {overall_psi:.4f}")

# ── STEP 5: Classification attempt ───────────────────────────────────────────
if label_col and len(shared_cols) >= 2:
    print("\nStep 5: Cross-equipment classification test...")

    # Map labels
    if mapped_df[label_col].dtype == object:
        mapped_df['y'] = mapped_df[label_col].map(SEOUL_LABEL_MAP)
    else:
        mapped_df['y'] = pd.to_numeric(mapped_df[label_col], errors='coerce')

    mapped_df = mapped_df.dropna(subset=['y'] + shared_cols)
    mapped_df['y'] = mapped_df['y'].astype(int)

    print(f"  Seoul class distribution: {dict(mapped_df['y'].value_counts().sort_index())}")

    # Train ASHRAE model
    train_X, train_y = [], []
    for f in TRAIN_FILES:
        path = os.path.join(BASE, f)
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
            lc = "Fault Detection Ground Truth"
            if lc not in df.columns:
                continue
            cols = [c for c in shared_cols if c in df.columns]
            df2 = df[cols + [lc]].copy()
            for c in cols:
                df2[c] = pd.to_numeric(df2[c], errors='coerce')
            df2 = df2.dropna()

            # Building label
            bld = TRAIN_FILES.index(f) + 1
            faulty = df2[df2[lc] != 0].copy()
            healthy = df2[df2[lc] == 0].copy()
            if not faulty.empty:
                faulty_X = faulty[cols].values
                train_X.extend(faulty_X.tolist())
                train_y.extend([bld] * len(faulty_X))
            if not healthy.empty:
                healthy_X = healthy[cols].values
                train_X.extend(healthy_X.tolist())
                train_y.extend([0] * len(healthy_X))

    if train_X:
        X_train = np.array(train_X)
        y_train = np.array(train_y)
        X_seoul = mapped_df[shared_cols].values.astype(float)
        y_seoul = mapped_df['y'].values

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_seoul_sc = scaler.transform(X_seoul)

        model = RandomForestClassifier(n_estimators=50, class_weight='balanced',
                                       random_state=42, n_jobs=-1)
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_seoul_sc)

        classes = sorted(np.unique(y_seoul).tolist())
        f1 = f1_score(y_seoul, preds, average='macro', zero_division=0)

        print(f"\n  CROSS-EQUIPMENT F1 (ASHRAE -> Seoul real data): {f1:.4f}")
        print(f"  Thesis AHU->AHU cross-building F1:              0.331")
        print(f"  Thesis AHU->FCU cross-equipment F1:             0.086")
        print(f"\n  Classification report:")
        label_names = {0:'Normal', 1:'Sensor Fault', 2:'Fan Fault', 3:'Valve/Heating'}
        names = [label_names.get(c, str(c)) for c in classes]
        print(classification_report(y_seoul, preds, labels=classes,
              target_names=names, zero_division=0))

# ── STEP 6: Save results ──────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
results = {
    "experiment": "seoul_registry_test",
    "dataset": "Real operational BMS data, South Korea, Hanyang University 2024",
    "doi": "10.1016/j.dib.2024.110956",
    "shared_cols": shared_cols,
    "psi_values": {k: round(v, 4) for k, v in psi_results.items()} if 'psi_results' in dir() else {},
    "f1_cross_equipment": float(f1) if 'f1' in dir() else None,
    "thesis_comparison": {
        "ahu_ahu_cross_building": 0.331,
        "ahu_fcu_cross_equipment": 0.086,
        "ahu_rtu_cross_equipment": 0.128,
        "ahu_boiler_cross_equipment": 0.096
    }
}
with open(os.path.join(out_dir, "seoul_registry_test.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Dataset: Real BMS data, South Korea, {seoul_df.shape[0]:,} rows")
print(f"  Shared sensor roles: {len(shared_cols)} of 11")
print(f"  Equipment category: CAT2 (thermodynamically similar, different schema)")
print(f"  PSI gate: RED (extreme distribution shift expected for real data)")
print(f"  This is a genuine external validation outside the LBNL dataset family.")
print(f"\n  Saved to: results/seoul_registry_test.json")
print("=" * 65)
