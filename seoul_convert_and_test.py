# seoul_convert_and_test.py
# Tests Universal FDD Registry on Seoul real BMS dataset
# Run: python seoul_convert_and_test.py

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report

BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
XLS_PATH  = os.path.join(BASE, "seoul", "combined_FDD.xls")
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

print("=" * 65)
print("SEOUL REAL BMS DATASET - UNIVERSAL REGISTRY TEST")
print("File: combined_FDD.xls")
print("=" * 65)

# ── STEP 1: Load xls ──────────────────────────────────────────────────────────
print(f"\nStep 1: Loading {XLS_PATH}...")
if not os.path.exists(XLS_PATH):
    print(f"  ERROR: File not found at {XLS_PATH}")
    exit(1)

try:
    xl = pd.ExcelFile(XLS_PATH, engine='xlrd')
except Exception:
    try:
        xl = pd.ExcelFile(XLS_PATH, engine='openpyxl')
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        print("  Try installing xlrd: pip install xlrd")
        exit(1)

print(f"  Sheets: {xl.sheet_names}")

all_frames = []
for sheet in xl.sheet_names:
    try:
        df_s = pd.read_excel(XLS_PATH, sheet_name=sheet, engine='xlrd')
    except Exception:
        df_s = pd.read_excel(XLS_PATH, sheet_name=sheet, engine='openpyxl')
    df_s['_sheet'] = sheet
    all_frames.append(df_s)
    print(f"  Sheet '{sheet}': {len(df_s):,} rows")

seoul_df = pd.concat(all_frames, ignore_index=True)
seoul_df.columns = seoul_df.columns.str.strip()
print(f"\n  Combined: {len(seoul_df):,} rows")
print(f"  Columns: {list(seoul_df.columns)}")

# ── STEP 2: Show sample data ──────────────────────────────────────────────────
print("\nStep 2: Sample data (first 3 rows):")
print(seoul_df.head(3).to_string())

# ── STEP 3: Find label column ─────────────────────────────────────────────────
print("\nStep 3: Finding label column...")
label_col = None
for possible in ["Labeling", "Label", "label", "labeling", "Fault", "fault",
                 "Class", "class", "Condition", "Status", "FaultType",
                 "fault_type", "Category", "Type", "M", "Labelling"]:
    if possible in seoul_df.columns:
        label_col = possible
        break

if label_col:
    print(f"  Label column: '{label_col}'")
    print(f"  Unique values: {seoul_df[label_col].dropna().unique().tolist()}")
    print(f"  Value counts:\n{seoul_df[label_col].value_counts()}")
else:
    print("  Label column not found automatically.")
    print("  All columns with sample values:")
    for col in seoul_df.columns:
        if col == '_sheet':
            continue
        sample = seoul_df[col].dropna().head(2).tolist()
        print(f"    '{col}': {seoul_df[col].dtype} — {sample}")

# ── STEP 4: Map columns ───────────────────────────────────────────────────────
print("\nStep 4: Mapping Seoul columns to ASHRAE equivalents...")

SEOUL_COL_CANDIDATES = {
    "AHU: Supply Air Temperature": [
        "Supply Air Temperature", "Supply Air Temp", "SA Temp",
        "SA_TEMP", "SAT", "D", "supply air temperature",
        "Supply_Air_Temperature", "SupplyAirTemp"
    ],
    "AHU: Outdoor Air Temperature": [
        "Ventilation Temperature", "Outdoor Air Temperature",
        "OA_TEMP", "OAT", "C", "ventilation temperature",
        "Outside Air Temperature", "Outdoor Temp", "Ventilation_Temperature"
    ],
    "AHU: Supply Air Fan Status": [
        "Supply Fan", "Fan Status", "Fan", "E", "supply fan",
        "SF_SPD_DM", "Fan_Status", "SupplyFan", "Supply_Fan"
    ],
    "AHU: Cooling Coil Valve Control Signal": [
        "Valve Position", "Cooling Valve", "Valve", "F",
        "valve position", "CHWC_VLV_DM", "Valve_Position",
        "CoolingValve", "ValvePosition"
    ],
    "AHU: Return Air Temperature": [
        "Return Air Temperature", "Return Air Temp", "RA_TEMP",
        "RAT", "Return_Air_Temperature", "ReturnAirTemp"
    ],
}

mapped_df = seoul_df.copy()
shared_cols = []

for ashrae_col, candidates in SEOUL_COL_CANDIDATES.items():
    found = False
    for cand in candidates:
        if cand in mapped_df.columns:
            mapped_df[ashrae_col] = pd.to_numeric(mapped_df[cand], errors='coerce')
            print(f"  MAPPED: '{cand}' -> '{ashrae_col}'")
            shared_cols.append(ashrae_col)
            found = True
            break
    if not found:
        print(f"  NOT FOUND: {ashrae_col}")

print(f"\n  Total shared columns: {len(shared_cols)}")

# ── STEP 5: PSI analysis ──────────────────────────────────────────────────────
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

psi_results = {}
if len(shared_cols) > 0:
    print("\nStep 5: PSI analysis (Seoul real data vs ASHRAE training)...")
    train_frames = []
    for f in TRAIN_FILES:
        path = os.path.join(BASE, f)
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
            cols_avail = [c for c in shared_cols if c in df.columns]
            for c in cols_avail:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            train_frames.append(df[cols_avail])

    train_ref = pd.concat(train_frames, ignore_index=True)
    print(f"  ASHRAE training reference: {len(train_ref):,} rows")

    print(f"\n  {'Sensor':50}  {'PSI':8}  Gate")
    print("  " + "-" * 68)
    for col in shared_cols:
        if col in train_ref.columns and col in mapped_df.columns:
            psi = compute_psi(train_ref[col], mapped_df[col])
            psi_results[col] = psi
            gate = "RED - ADAPTATION REQUIRED" if psi > 0.5 else "AMBER" if psi > 0.2 else "GREEN"
            print(f"  {col:50}  {psi:.4f}   {gate}")

# ── STEP 6: Classification ────────────────────────────────────────────────────
LABEL_MAP = {
    "Normal": 0, "normal": 0, "Normal condition": 0, "0": 0,
    "Sensor fault": 1, "sensor fault": 1, "Sensor Fault": 1,
    "Return air temperature sensor fault": 1,
    "Supply air Temperature sensor fault": 1,
    "Supply air temperature sensor fault": 1,
    "Supply Air Temperature sensor fault": 1,
    "Fan fault": 2, "fan fault": 2, "Fan Fault": 2,
    "Supply fan fault": 2, "Total Heating Pump": 3,
    "Valve fault": 3, "valve fault": 3, "Valve Fault": 3,
    "total heating pump fault": 3, "Total heating pump fault": 3,
}

f1 = None
if label_col and len(shared_cols) >= 2:
    print("\nStep 6: Classification (ASHRAE model on Seoul real data)...")
    if mapped_df[label_col].dtype == object:
        mapped_df['y'] = mapped_df[label_col].map(LABEL_MAP)
    else:
        mapped_df['y'] = pd.to_numeric(mapped_df[label_col], errors='coerce')

    unmapped = mapped_df[mapped_df['y'].isna()][label_col].unique()
    if len(unmapped) > 0:
        print(f"  Unmapped labels: {unmapped[:5]}")

    mapped_df = mapped_df.dropna(subset=['y'] + shared_cols)
    mapped_df['y'] = mapped_df['y'].astype(int)
    print(f"  Usable rows: {len(mapped_df):,}")
    print(f"  Class distribution: {dict(mapped_df['y'].value_counts().sort_index())}")

    # Build ASHRAE training
    train_X, train_y = [], []
    for i, f_name in enumerate(TRAIN_FILES):
        path = os.path.join(BASE, f_name)
        if not os.path.exists(path):
            continue
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
        faulty  = df2[df2[lc] != 0]
        healthy = df2[df2[lc] == 0]
        if not faulty.empty:
            train_X.extend(faulty[cols].values.tolist())
            train_y.extend([i+1] * len(faulty))
        if not healthy.empty:
            train_X.extend(healthy[cols].values.tolist())
            train_y.extend([0] * len(healthy))

    if train_X:
        X_train = np.array(train_X)
        y_train = np.array(train_y)
        X_seoul = mapped_df[shared_cols].values.astype(float)
        y_seoul = mapped_df['y'].values

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_seoul_sc = scaler.transform(X_seoul)

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                       random_state=42, n_jobs=-1)
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_seoul_sc)
        f1 = f1_score(y_seoul, preds, average='macro', zero_division=0)

        print(f"\n  RESULT: ASHRAE model on Seoul real data F1 = {f1:.4f}")
        print(f"\n  Comparison table:")
        print(f"    AHU -> AHU (same equipment, unseen building): 0.331")
        print(f"    AHU -> FCU (similar equipment):               0.086")
        print(f"    AHU -> RTU (architectural mismatch):          0.128")
        print(f"    AHU -> Boiler (architectural mismatch):       0.096")
        print(f"    AHU -> Seoul (REAL Korean BMS data):          {f1:.4f}")

        classes = sorted(np.unique(y_seoul).tolist())
        lnames = {0:'Normal', 1:'Sensor', 2:'Fan', 3:'Valve/Pump'}
        print(f"\n  Classification report:")
        print(classification_report(y_seoul, preds,
              labels=classes,
              target_names=[lnames.get(c, str(c)) for c in classes],
              zero_division=0))

# ── Save results ──────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
output = {
    "experiment": "seoul_real_bms_registry_test",
    "dataset": "Real BMS, South Korea, Hanyang University 2024",
    "doi": "10.1016/j.dib.2024.110956",
    "rows": int(len(seoul_df)),
    "columns_in_file": list(seoul_df.columns),
    "shared_cols_mapped": shared_cols,
    "psi_values": {k: round(v,4) for k,v in psi_results.items()},
    "f1_cross_equipment": float(f1) if f1 is not None else None,
    "thesis_comparison": {
        "ahu_ahu_cross_building": 0.331,
        "ahu_fcu": 0.086,
        "ahu_rtu": 0.128,
        "ahu_boiler": 0.096,
    }
}
with open(os.path.join(out_dir, "seoul_registry_test.json"), "w") as fout:
    json.dump(output, fout, indent=2)

print("\n" + "=" * 65)
print("DONE - Results saved to results/seoul_registry_test.json")
print("=" * 65)
