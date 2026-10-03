# FIX 6+7 - SD-AHU PSI values and Physics Normalisation PSI before/after
# Run: python fix6_sdahu_physics_psi.py

import warnings, os, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SDAHU_DIR = os.path.join(BASE, "sdahu")
FCU_DIR   = os.path.join(BASE, "fcu")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

SDAHU_COL_MAP = {
    "SF_SPD":   "AHU: Supply Air Fan Speed Control Signal",
    "OA_TEMP":  "AHU: Outdoor Air Temperature",
    "RA_TEMP":  "AHU: Return Air Temperature",
    "OA_DMPR":  "AHU: Outdoor Air Damper Control Signal",
    "MA_TEMP":  "AHU: Mixed Air Temperature",
    "SA_TEMP":  "AHU: Supply Air Temperature",
    "CHWC_VLV": "AHU: Cooling Coil Valve Control Signal",
}

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

PHYS_RANGES = {
    "AHU: Outdoor Air Temperature":             (-20.0, 45.0),
    "AHU: Mixed Air Temperature":               (-5.0,  40.0),
    "AHU: Return Air Temperature":              (10.0,  35.0),
    "AHU: Supply Air Temperature":              (5.0,   30.0),
    "AHU: Supply Air Fan Speed Control Signal": (0.0,   1.0),
    "AHU: Outdoor Air Damper Control Signal":   (0.0,   1.0),
    "AHU: Cooling Coil Valve Control Signal":   (0.0,   1.0),
    "AHU: Heating Coil Valve Control Signal":   (0.0,   1.0),
}

def load_clean(path, cap=15000):
    df = pd.read_csv(path, nrows=cap, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_psi(ref_vals, dep_vals, n_bins=10):
    ref_clean = ref_vals.dropna().astype(float)
    dep_clean = dep_vals.dropna().astype(float)
    if len(ref_clean) == 0 or len(dep_clean) == 0:
        return 0.0
    mn = float(min(ref_clean.min(), dep_clean.min()))
    mx = float(max(ref_clean.max(), dep_clean.max()))
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    eps  = 1e-6
    ref_c = np.histogram(ref_clean, bins=bins)[0].astype(float) + eps
    dep_c = np.histogram(dep_clean, bins=bins)[0].astype(float) + eps
    ref_p = ref_c / ref_c.sum()
    dep_p = dep_c / dep_c.sum()
    return float(np.sum((dep_p - ref_p) * np.log(dep_p / ref_p)))

# Build training reference
print("Loading training reference data...")
train_frames = []
for f in ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]:
    df = load_clean(os.path.join(BASE, f), cap=50000)
    train_frames.append(df)
train_ref = pd.concat(train_frames, ignore_index=True)
print(f"  Training reference: {len(train_ref):,} rows")

# ── FIX 6: SD-AHU PSI ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FIX 6 - SD-AHU PSI VALUES")
print("=" * 65)

sdahu_path = os.path.join(SDAHU_DIR, "AHU_annual.csv")
if os.path.exists(sdahu_path):
    sdahu_df = load_clean(sdahu_path, cap=15000)
    sdahu_df = sdahu_df.rename(columns=SDAHU_COL_MAP)

    psi_sdahu = {}
    for ahu_col in SDAHU_COL_MAP.values():
        if ahu_col in train_ref.columns and ahu_col in sdahu_df.columns:
            psi_sdahu[ahu_col] = compute_psi(train_ref[ahu_col], sdahu_df[ahu_col])

    psi_sorted = sorted(psi_sdahu.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Sensor':50} {'PSI':8}  Category")
    print("  " + "-" * 65)
    for col, psi in psi_sorted:
        cat = "Extreme" if psi > 0.5 else "High" if psi > 0.2 else "Moderate" if psi > 0.1 else "Low"
        print(f"  {col:50}  {psi:.4f}   {cat}")
else:
    print(f"  File not found: {sdahu_path}")

# ── FIX 7: Physics Normalisation PSI Before vs After ──────────────────────────
print("\n" + "=" * 75)
print("FIX 7 - PHYSICS NORMALISATION PSI BEFORE vs AFTER")
print("=" * 75)

fcu_path = os.path.join(FCU_DIR, "FCU_FaultFree.csv")
if os.path.exists(fcu_path):
    fcu_df = load_clean(fcu_path, cap=15000)
    fcu_df = fcu_df.rename(columns=FCU_COL_MAP)

    shared = [c for c in FCU_COL_MAP.values()
              if c in train_ref.columns and c in fcu_df.columns]

    print(f"  {'Sensor':50} {'Before':8}  {'After':8}  Change")
    print("  " + "-" * 75)

    psi_before_after = {}
    for col in shared:
        psi_before = compute_psi(train_ref[col], fcu_df[col])

        if col in PHYS_RANGES:
            lo, hi = PHYS_RANGES[col]
            margin = 0.1 * (hi - lo)
            ref_n = ((train_ref[col].clip(lo - margin, hi + margin) - lo) / (hi - lo))
            dep_n = ((fcu_df[col].clip(lo - margin, hi + margin) - lo) / (hi - lo))
            psi_after = compute_psi(ref_n, dep_n)
        else:
            psi_after = psi_before

        psi_before_after[col] = {"before": round(psi_before, 4), "after": round(psi_after, 4)}
        change = "Reduced" if psi_after < psi_before else "INCREASED"
        print(f"  {col:50}  {psi_before:6.4f}    {psi_after:6.4f}    {change}")
else:
    print(f"  File not found: {fcu_path}")

# Save
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out = {
    "sdahu_psi":          dict(psi_sorted) if "psi_sorted" in dir() else {},
    "physics_norm_psi":   psi_before_after if "psi_before_after" in dir() else {},
}
with open(os.path.join(out_dir, "psi_fixes.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to results/psi_fixes.json")
print("\nCopy these values into Tables 4.5 and 4.10 in Chapter 4.")
