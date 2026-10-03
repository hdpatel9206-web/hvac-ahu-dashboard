# wang_multibuilding_test.py - FIXED VERSION
# Wang et al. 2025 multi-building real BMS validation
# Run: python wang_multibuilding_test.py

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report

BASE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WANG_DIR     = os.path.join(BASE, "wang")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
TRAIN_FILES  = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]

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

# EXACT column names confirmed from the output above
WANG_COL_MAP = {
    "Supply air temperature":  "AHU: Supply Air Temperature",
    "Return temperature":      "AHU: Return Air Temperature",
    "Supply fan":              "AHU: Supply Air Fan Status",
    "Valve position":          "AHU: Cooling Coil Valve Control Signal",
}

BUILDING_FILES = {
    "Office":     "office_scientific_data.csv",
    "Auditorium": "auditorium_scientific_data.csv",
    "Hospital":   "hosptial_scientific_data.csv",
}

# Label mapping using string matching
LABEL_MAP_RULES = {
    0: ["normal condition", "normal", "healthy", "no fault"],
    1: ["supply air temperature fault", "return air temperature fault",
        "temperature sensor", "sensor fault"],
    2: ["supply fan fault", "fan fault"],
    3: ["valve position fault", "valve fault", "heating pump fault",
        "cooling pump fault", "pump fault"],
}

def map_label(val):
    if pd.isna(val):
        return -1
    v = str(val).lower().strip()
    for label_num, keywords in LABEL_MAP_RULES.items():
        for kw in keywords:
            if kw in v:
                return label_num
    return -1

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

print("=" * 70)
print("WANG ET AL. 2025 - MULTI-BUILDING REAL BMS VALIDATION")
print("Office | Auditorium | Hospital (South Korea)")
print("DOI: 10.6084/m9.figshare.27147678.v3")
print("=" * 70)

# Load ASHRAE training reference
print("\nLoading ASHRAE training reference...")
train_frames = []
for fname in TRAIN_FILES:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for c in ASHRAE_RAW_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    avail = [c for c in ASHRAE_RAW_COLS if c in df.columns]
    train_frames.append(df[avail])
    print(f"  {fname}: {len(df):,} rows")

train_ref = pd.concat(train_frames, ignore_index=True)
print(f"  Total: {len(train_ref):,} rows")

all_results = {}

for building_type, filename in BUILDING_FILES.items():
    print(f"\n{'=' * 70}")
    print(f"BUILDING: {building_type.upper()}")
    print(f"{'=' * 70}")

    filepath = os.path.join(WANG_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  NOT FOUND: {filepath}")
        continue

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")

    # Apply exact column mapping
    shared_cols = []
    for wang_col, ashrae_col in WANG_COL_MAP.items():
        if wang_col in df.columns:
            df[ashrae_col] = pd.to_numeric(df[wang_col], errors='coerce')
            shared_cols.append(ashrae_col)
            print(f"  MAPPED: '{wang_col}' -> '{ashrae_col}'")
        else:
            print(f"  NOT FOUND: '{wang_col}'")

    print(f"\n  Shared columns: {len(shared_cols)}")

    # PSI analysis
    print(f"\n  PSI Analysis ({building_type} vs ASHRAE training):")
    print(f"  {'Sensor':50}  {'PSI':8}  Gate  | ASHRAE CB PSI")
    print("  " + "-" * 75)

    ashrae_ref_psi = {
        "AHU: Supply Air Temperature":           2.94,
        "AHU: Return Air Temperature":           0.73,
        "AHU: Supply Air Fan Status":            0.28,
        "AHU: Cooling Coil Valve Control Signal":0.88,
    }

    psi_results = {}
    for col in shared_cols:
        if col in train_ref.columns and col in df.columns:
            psi = compute_psi(train_ref[col], df[col])
            psi_results[col] = psi
            gate = "RED" if psi > 0.5 else "AMBER" if psi > 0.2 else "GREEN"
            ref_psi = ashrae_ref_psi.get(col, "-")
            print(f"  {col:50}  {psi:8.4f}  {gate:5}  | {ref_psi}")

    # Label mapping
    label_col = 'labeling'
    df['y'] = df[label_col].apply(map_label)
    unmapped = df[df['y'] == -1][label_col].unique()
    if len(unmapped) > 0:
        print(f"\n  Unmapped labels (assigned -1): {unmapped.tolist()}")

    df_clean = df[df['y'] >= 0].dropna(subset=shared_cols).copy()
    df_clean['y'] = df_clean['y'].astype(int)
    print(f"\n  Class distribution (after mapping):")
    label_names = {0:'Normal', 1:'Sensor Fault', 2:'Fan Fault', 3:'Valve/Pump'}
    for cls, cnt in df_clean['y'].value_counts().sort_index().items():
        print(f"    Class {cls} ({label_names.get(cls,'?')}): {cnt:,} rows")

    # Classification
    f1 = None
    if len(df_clean) > 10 and len(df_clean['y'].unique()) >= 2 and len(shared_cols) >= 2:
        tX, ty = [], []
        for i, fname in enumerate(TRAIN_FILES):
            path = os.path.join(BASE, fname)
            if not os.path.exists(path):
                continue
            adf = pd.read_csv(path, low_memory=False)
            adf.columns = adf.columns.str.strip()
            adf.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
            lc = "Fault Detection Ground Truth"
            if lc not in adf.columns:
                continue
            cols = [c for c in shared_cols if c in adf.columns]
            adf2 = adf[cols + [lc]].copy()
            for c in cols:
                adf2[c] = pd.to_numeric(adf2[c], errors='coerce')
            adf2 = adf2.dropna()
            faulty  = adf2[adf2[lc] != 0]
            healthy = adf2[adf2[lc] == 0]
            if not faulty.empty:
                tX.extend(faulty[cols].values.tolist())
                ty.extend([i+1] * len(faulty))
            if not healthy.empty:
                tX.extend(healthy[cols].values.tolist())
                ty.extend([0] * len(healthy))

        if tX:
            X_train = np.array(tX).astype(float)
            y_train = np.array(ty).astype(int)
            X_test  = df_clean[shared_cols].values.astype(float)
            y_test  = df_clean['y'].values.astype(int)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100,
                    class_weight='balanced', random_state=42, n_jobs=-1)
            model.fit(X_tr_sc, y_train)
            preds = model.predict(X_te_sc)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            classes = sorted(np.unique(y_test).tolist())
            names   = [label_names.get(c, str(c)) for c in classes]
            print(f"\n  RESULT: ASHRAE -> {building_type} F1 = {f1:.4f}")
            print(classification_report(y_test, preds,
                  labels=classes, target_names=names, zero_division=0))

    all_results[building_type] = {
        "rows":       int(len(df)),
        "shared_cols": shared_cols,
        "psi_values": {k: round(v,4) for k,v in psi_results.items()},
        "f1":         round(float(f1),4) if f1 is not None else None,
    }

# Final summary
print("\n" + "=" * 70)
print("COMPLETE EXTERNAL VALIDATION SUMMARY")
print("=" * 70)
print(f"\n  {'Building':15}  {'Rows':8}  {'Shared':8}  {'Max PSI':10}  {'F1':8}  Gate")
print("  " + "-" * 65)

for btype, res in all_results.items():
    max_psi = max(res['psi_values'].values()) if res['psi_values'] else 0.0
    f1_s    = f"{res['f1']:.4f}" if res['f1'] is not None else "N/A"
    gate    = "RED" if max_psi > 0.5 else "AMBER" if max_psi > 0.2 else "GREEN"
    print(f"  {btype:15}  {res['rows']:8,}  {len(res['shared_cols']):4} cols  "
          f"{max_psi:10.2f}  {f1_s:8}  {gate}")

print(f"  {'Seoul':15}  {'5,471':8}  {'4 cols':8}  {'40.25':>10}  {'0.1829':8}  RED")
print(f"\n  ASHRAE benchmark (simulation):")
print(f"    AHU->AHU cross-building:     F1=0.331  Max PSI= 2.63  CAT1")
print(f"    AHU->FCU cross-equipment:    F1=0.086  Max PSI= 8.32  CAT2")
print(f"    AHU->RTU cross-equipment:    F1=0.128  Max PSI= 0.08  CAT3")
print(f"    AHU->Boiler cross-equipment: F1=0.096  Max PSI= 4.00  CAT3")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "wang_multibuilding_test.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to results/wang_multibuilding_test.json")
