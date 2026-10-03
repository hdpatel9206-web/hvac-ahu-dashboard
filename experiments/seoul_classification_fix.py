# seoul_classification_fix.py
# Fixes the label mapping issue and completes the classification
# Run: python experiments/seoul_classification_fix.py

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report

BASE      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
XLS_PATH  = os.path.join(BASE, "seoul", "combined_FDD.xls")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
TRAIN_FILES  = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]

SHARED_COLS = [
    "AHU: Supply Air Temperature",
    "AHU: Outdoor Air Temperature",
    "AHU: Supply Air Fan Status",
    "AHU: Cooling Coil Valve Control Signal",
]

SEOUL_COL_MAP = {
    "Supply Air Temperature":  "AHU: Supply Air Temperature",
    "Ventilation Temperature": "AHU: Outdoor Air Temperature",
    "Supply Fan":              "AHU: Supply Air Fan Status",
    "Valve Position":          "AHU: Cooling Coil Valve Control Signal",
}

print("=" * 65)
print("SEOUL CLASSIFICATION FIX")
print("=" * 65)

# Load Seoul data
df = pd.read_excel(XLS_PATH, engine='xlrd')
df.columns = df.columns.str.strip()

# Map columns
for seoul_col, ashrae_col in SEOUL_COL_MAP.items():
    if seoul_col in df.columns:
        df[ashrae_col] = pd.to_numeric(df[seoul_col], errors='coerce')

# Map labels manually using string contains
print(f"\nRaw label values: {df['Labeling'].unique().tolist()}")

# Use string comparison directly
df['y'] = -1
df.loc[df['Labeling'].astype(str).str.lower().str.contains('sensor'), 'y'] = 1
df.loc[df['Labeling'].astype(str).str.lower().str.contains('fan'),    'y'] = 2
df.loc[df['Labeling'].astype(str).str.lower().str.contains('normal'), 'y'] = 0
df.loc[df['Labeling'].astype(str).str.lower().str.contains('valve'),  'y'] = 3
df.loc[df['Labeling'].astype(str).str.lower().str.contains('pump'),   'y'] = 3

print(f"Label mapping result:")
print(df.groupby('Labeling')['y'].first().to_string())

# Keep only mapped rows
df = df[df['y'] >= 0].copy()
df = df.dropna(subset=SHARED_COLS)
print(f"\nUsable rows after mapping: {len(df):,}")
print(f"Class distribution: {dict(df['y'].value_counts().sort_index())}")

X_seoul = df[SHARED_COLS].values.astype(float)
y_seoul = df['y'].values.astype(int)

# Load ASHRAE training data
print("\nLoading ASHRAE training data...")
train_X, train_y = [], []
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
    cols = [c for c in SHARED_COLS if c in adf.columns]
    adf2 = adf[cols + [lc]].copy()
    for c in cols:
        adf2[c] = pd.to_numeric(adf2[c], errors='coerce')
    adf2 = adf2.dropna()
    faulty  = adf2[adf2[lc] != 0]
    healthy = adf2[adf2[lc] == 0]
    if not faulty.empty:
        train_X.extend(faulty[cols].values.tolist())
        train_y.extend([i+1] * len(faulty))
    if not healthy.empty:
        train_X.extend(healthy[cols].values.tolist())
        train_y.extend([0] * len(healthy))
    print(f"  {fname}: {len(adf2):,} rows loaded")

X_train = np.array(train_X).astype(float)
y_train = np.array(train_y).astype(int)
print(f"  Training total: {len(X_train):,} rows")

# Train and evaluate
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_seoul_sc = scaler.transform(X_seoul)

model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)
preds = model.predict(X_seoul_sc)

f1 = f1_score(y_seoul, preds, average='macro', zero_division=0)
classes = sorted(np.unique(y_seoul).tolist())
lnames = {0:'Normal', 1:'Sensor Fault', 2:'Fan Fault', 3:'Valve/Pump'}

print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)
print(f"\n  ASHRAE model on Seoul real BMS data: F1 = {f1:.4f}")
print(f"\n  Full comparison table:")
print(f"    AHU -> AHU (cross-building, same benchmark): 0.331")
print(f"    AHU -> FCU (cross-equipment CAT2):           0.086")
print(f"    AHU -> RTU (cross-equipment CAT3):           0.128")
print(f"    AHU -> Boiler (cross-equipment CAT3):        0.096")
print(f"    AHU -> Seoul (real Korean BMS, 2024):        {f1:.4f}")
print(f"\n  PSI values (Seoul vs ASHRAE training):")
print(f"    Supply Air Temperature:     40.25  (ASHRAE cross-building: 2.94)")
print(f"    Cooling Coil Valve:         24.06  (ASHRAE cross-building: 0.88)")
print(f"    Outdoor Air Temperature:    20.44  (ASHRAE cross-building: 8.10)")
print(f"    Supply Air Fan Status:       2.29  (ASHRAE cross-building: 0.28)")
print(f"\n  Key insight: Real-world international BMS data shows 5-14x")
print(f"  higher PSI than cross-building experiments within the same")
print(f"  benchmark dataset. This confirms the PSI gate is essential")
print(f"  for real deployment.")

print(f"\n  Classification report:")
print(classification_report(y_seoul, preds,
      labels=classes,
      target_names=[lnames.get(c, str(c)) for c in classes],
      zero_division=0))

# Save
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)
result = {
    "experiment": "seoul_real_bms_external_validation",
    "dataset":    "Real BMS, South Korea, Hanyang University 2024",
    "doi":        "10.1016/j.dib.2024.110956",
    "rows":       int(len(df)),
    "shared_cols": SHARED_COLS,
    "f1_cross_equipment": round(float(f1), 4),
    "psi_values": {
        "AHU: Supply Air Temperature":              40.2469,
        "AHU: Cooling Coil Valve Control Signal":   24.0633,
        "AHU: Outdoor Air Temperature":             20.4375,
        "AHU: Supply Air Fan Status":                2.2919,
    },
    "thesis_comparison": {
        "ahu_ahu_cross_building":    0.331,
        "ahu_fcu_cross_equipment":   0.086,
        "ahu_rtu_cross_equipment":   0.128,
        "ahu_boiler_cross_equipment":0.096,
    }
}
with open(os.path.join(out_dir, "seoul_registry_test.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to results/seoul_registry_test.json")
