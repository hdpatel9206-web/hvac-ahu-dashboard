# cork_psi_test.py - FIXED VERSION
# Direct path to Cork file - no auto-detection
# Run: python experiments/cork_psi_test.py

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_FILES  = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

# DIRECT PATH - no auto-detection
CORK_PATH = os.path.join(BASE, "cork", "Data_Article_Dataset.csv")

CORK_COL_MAP = {
    "RaTemp":    "AHU: Return Air Temperature",
    "OaTemp":    "AHU: Outdoor Air Temperature",
    "MaTemp":    "AHU: Mixed Air Temperature",
    "OaDmprPos": "AHU: Outdoor Air Damper Control Signal",
    "RaDmprPos": "AHU: Return Air Damper Control Signal",
    "HWVlvPos":  "AHU: Heating Coil Valve Control Signal",
    "ChWVlvPos": "AHU: Cooling Coil Valve Control Signal",
    "DaTemp":    "AHU: Supply Air Temperature",
}

ASHRAE_CB_PSI = {
    "AHU: Return Air Temperature":             0.73,
    "AHU: Outdoor Air Temperature":            8.10,
    "AHU: Mixed Air Temperature":              0.85,
    "AHU: Outdoor Air Damper Control Signal":  0.59,
    "AHU: Return Air Damper Control Signal":   1.99,
    "AHU: Heating Coil Valve Control Signal":  0.49,
    "AHU: Cooling Coil Valve Control Signal":  0.88,
    "AHU: Supply Air Temperature":             2.94,
}

WANG_OFFICE_PSI = {
    "AHU: Supply Air Temperature":             43.48,
    "AHU: Return Air Temperature":             50.66,
    "AHU: Cooling Coil Valve Control Signal":  11.99,
}

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
print("CORK INDUSTRIAL AHU - PSI ANALYSIS")
print("Ahern, O'Sullivan, Bruton (2023) | University College Cork, Ireland")
print("DOI: 10.17632/8x62ntvrg7.2")
print("=" * 70)

print(f"\nLoading: {CORK_PATH}")
if not os.path.exists(CORK_PATH):
    print(f"  ERROR: File not found at {CORK_PATH}")
    exit(1)

cork_df = pd.read_csv(CORK_PATH, low_memory=False)
cork_df.columns = cork_df.columns.str.strip()
print(f"  Shape: {cork_df.shape}")
print(f"  Columns: {list(cork_df.columns)}")

# Show date range if datetime column exists
time_col = None
for c in ['Datetime', 'datetime', 'Date', 'date', 'Time', 'time', 'Timestamp']:
    if c in cork_df.columns:
        time_col = c
        break
if time_col:
    print(f"  Date range: {cork_df[time_col].iloc[0]} to {cork_df[time_col].iloc[-1]}")

# Map columns
print("\nMapping columns...")
shared_cols = []
for cork_col, ashrae_col in CORK_COL_MAP.items():
    if cork_col in cork_df.columns:
        cork_df[ashrae_col] = pd.to_numeric(cork_df[cork_col], errors='coerce')
        non_null = cork_df[ashrae_col].notna().sum()
        pct = 100 * non_null / len(cork_df)
        print(f"  MAPPED: '{cork_col}' -> '{ashrae_col}' ({non_null:,} rows, {pct:.0f}% non-null)")
        shared_cols.append(ashrae_col)
    else:
        print(f"  NOT FOUND: '{cork_col}'")

print(f"\n  Shared columns: {len(shared_cols)}")

# Load ASHRAE training reference
print("\nLoading ASHRAE training reference...")
train_frames = []
for fname in TRAIN_FILES:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, nrows=50000, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    avail = [c for c in shared_cols if c in df.columns]
    for c in avail:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    train_frames.append(df[avail])
    print(f"  {fname}: loaded")

train_ref = pd.concat(train_frames, ignore_index=True)

# PSI analysis
print("\n" + "=" * 80)
print("PSI RESULTS: Cork Ireland vs ASHRAE Training")
print(f"{'Sensor':48}  {'Cork':8}  {'ASHRAE CB':10}  {'Wang Office':12}  Gate")
print("-" * 80)

psi_results = {}
for col in shared_cols:
    if col in train_ref.columns and col in cork_df.columns:
        psi = compute_psi(train_ref[col], cork_df[col])
        psi_results[col] = psi
        gate = "RED" if psi > 0.5 else "AMBER" if psi > 0.2 else "GREEN"
        a = str(ASHRAE_CB_PSI.get(col, "-"))
        w = str(WANG_OFFICE_PSI.get(col, "-"))
        print(f"{col:48}  {psi:8.4f}  {a:>10}  {w:>12}  {gate}")

max_psi = max(psi_results.values()) if psi_results else 0

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Dataset:        Real industrial BMS, Cork Ireland, 2015-2022")
print(f"  Rows:           {len(cork_df):,} at 15-min intervals")
print(f"  Shared cols:    {len(shared_cols)} of 11 (richest real-world mapping)")
print(f"  Maximum PSI:    {max_psi:.4f}")
print(f"  Gate:           {'RED' if max_psi > 0.5 else 'AMBER'}")
print(f"  F1:             Not computed (no fault labels)")
print(f"  Category:       CAT3 if RED, CAT2 if AMBER")
print(f"\n  Three-country PSI comparison:")
print(f"    US simulation (ASHRAE cross-building): max PSI ~2.94")
print(f"    Korea real BMS (Wang office):          max PSI 50.66")
print(f"    Ireland real industrial (Cork):        max PSI {max_psi:.2f}")

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)
result = {
    "experiment":  "cork_industrial_psi",
    "dataset":     "Real industrial BMS, Cork Ireland, 2015-2022",
    "doi":         "10.17632/8x62ntvrg7.2",
    "rows":        int(len(cork_df)),
    "shared_cols": shared_cols,
    "psi_values":  {k: round(v,4) for k,v in psi_results.items()},
    "max_psi":     round(max_psi, 4),
    "gate":        "RED" if max_psi > 0.5 else "AMBER",
    "has_labels":  False,
    "note":        "PSI only - no fault labels in Cork dataset"
}
with open(os.path.join(out_dir, "cork_psi_test.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to results/cork_psi_test.json")
