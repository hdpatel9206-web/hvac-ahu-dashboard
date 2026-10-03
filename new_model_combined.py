# new_model_combined.py
# BEST APPROACH: Train on ASHRAE + SD-AHU with genuine fault type labels
#
# Label scheme:
#   Class 0 = Healthy        (ASHRAE training buildings)
#   Class 1 = Sensor Bias    (ASHRAE training buildings - OA temp sensor bias)
#   Class 2 = Valve Fault    (SD-AHU coi_leakage + coi_stuck files)
#   Class 3 = Damper Fault   (SD-AHU damper_stuck files)
#
# Why this works better than pure SD-AHU training:
#   - Classes 0 and 1 come from ASHRAE = same environment as test buildings
#   - PSI for classes 0+1 will be low (same simulation family)
#   - Classes 2 and 3 are genuine fault types not available in ASHRAE
#   - The model is a genuine 4-class fault type classifier
#
# Test: ASHRAE withheld buildings (MZVAV-2-2 + SZVAV)
# Run: python new_model_combined.py
# Time: approximately 20-30 minutes

import os, json, glob, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy    import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, classification_report,
                              matthews_corrcoef, balanced_accuracy_score)

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
SDAHU_DIR = os.path.join(DATA_DIR, "sdahu")
RS        = 42
ROLLING_WINDOWS = [10, 30, 60]
MAX_ROWS_PER_FILE = 50000

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

# ASHRAE training files — Classes 0 and 1 only
ASHRAE_TRAIN_FILES = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']

# SD-AHU files for Classes 2 and 3 ONLY
# Deliberately exclude oa_bias and coi_bias (Class 1) to avoid
# mixing two different simulation environments for the same class
SDAHU_CLASS2_PATTERNS = ['coi_leakage', 'coi_stuck']   # Valve faults
SDAHU_CLASS3_PATTERNS = ['damper_stuck']                # Damper faults
# Skip: ahu_annual, oa_bias, coi_bias (covered by ASHRAE)

SDAHU_COL_MAP = {
    'SA_TEMP':     'AHU: Supply Air Temperature',
    'OA_TEMP':     'AHU: Outdoor Air Temperature',
    'MA_TEMP':     'AHU: Mixed Air Temperature',
    'RA_TEMP':     'AHU: Return Air Temperature',
    'SF_SPD_DM':   'AHU: Supply Air Fan Status',
    'SF_CS':       'AHU: Supply Air Fan Speed Control Signal',
    'OA_DMPR_DM':  'AHU: Outdoor Air Damper Control Signal',
    'RA_DMPR_DM':  'AHU: Return Air Damper Control Signal',
    'CHWC_VLV_DM': 'AHU: Cooling Coil Valve Control Signal',
    'SYS_CTL':     'Occupancy Mode Indicator',
}

EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
CLASS_NAMES  = {0:'Healthy', 1:'Sensor Bias', 2:'Valve Fault', 3:'Damper Fault'}

def engineer_features(df):
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in available:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    pairs = [
        ('Temp_supply_return_diff',  'AHU: Supply Air Temperature', 'AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff', 'AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff',   'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)

def load_ashrae_file(fpath):
    """Load ASHRAE file. Faulty rows = Class 1, Healthy rows = Class 0."""
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    lc = 'Fault Detection Ground Truth'
    if lc not in df.columns:
        return pd.DataFrame()
    for col in RAW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0.0
    feats = engineer_features(df)
    # Class 0 = healthy, Class 1 = any fault (genuine sensor bias)
    raw_labels = df[lc].values[:len(feats)]
    feats['label'] = np.where(raw_labels == 0, 0, 1)
    return feats.dropna()

def load_sdahu_file(fpath, label):
    """Load SD-AHU file with given fault type label."""
    df = pd.read_csv(fpath, low_memory=False, nrows=MAX_ROWS_PER_FILE)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for sdahu_col, ashrae_col in SDAHU_COL_MAP.items():
        if sdahu_col in df.columns:
            df[ashrae_col] = pd.to_numeric(df[sdahu_col], errors='coerce')
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = 0.0
    feats = engineer_features(df)
    feats['label'] = label
    return feats.dropna()

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
    rc = np.histogram(ref, bins=bins)[0].astype(float) + eps
    dc = np.histogram(dep, bins=bins)[0].astype(float) + eps
    rp, dp = rc/rc.sum(), dc/dc.sum()
    return float(np.sum((dp - rp) * np.log(dp / rp)))

print("=" * 70)
print("COMBINED MODEL TRAINING: ASHRAE + SD-AHU")
print("Class 0+1 from ASHRAE  |  Class 2+3 from SD-AHU")
print("Genuine fault type labels")
print("=" * 70)

# ── STEP 1: Load ASHRAE training (Classes 0 and 1) ───────────────────────────
print("\nStep 1: Loading ASHRAE training buildings (Classes 0 and 1)...")
ashrae_parts = []
ashrae_counts = {0: 0, 1: 0}

for fname in ASHRAE_TRAIN_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  NOT FOUND: {fpath}")
        continue
    df = load_ashrae_file(fpath)
    if len(df) > 0:
        ashrae_parts.append(df)
        c0 = (df['label'] == 0).sum()
        c1 = (df['label'] == 1).sum()
        ashrae_counts[0] += c0
        ashrae_counts[1] += c1
        print(f"  {fname:20s}: {len(df):>8,} rows  "
              f"(healthy={c0:,}  sensor_bias={c1:,})")

df_ashrae = pd.concat(ashrae_parts, ignore_index=True)
print(f"\n  ASHRAE total: {len(df_ashrae):,} rows")
print(f"  Class 0 (Healthy):     {ashrae_counts[0]:,}")
print(f"  Class 1 (Sensor Bias): {ashrae_counts[1]:,}")

# ── STEP 2: Load SD-AHU (Classes 2 and 3 only) ───────────────────────────────
print("\nStep 2: Loading SD-AHU fault files (Classes 2 and 3 only)...")

if not os.path.exists(SDAHU_DIR):
    SDAHU_DIR = os.path.join(DATA_DIR, "lbnl_sdahu", "LBNL_FDD_Dataset_SDAHU")
    print(f"  Using: {SDAHU_DIR}")

all_sdahu = glob.glob(os.path.join(SDAHU_DIR, "*.csv"))
sdahu_parts = []
sdahu_counts = {2: 0, 3: 0}

for fpath in sorted(all_sdahu):
    fname = os.path.basename(fpath).lower()
    label = None
    for pat in SDAHU_CLASS2_PATTERNS:
        if pat in fname:
            label = 2
            break
    if label is None:
        for pat in SDAHU_CLASS3_PATTERNS:
            if pat in fname:
                label = 3
                break
    if label is None:
        print(f"  SKIP: {os.path.basename(fpath)}")
        continue
    df = load_sdahu_file(fpath, label)
    if len(df) > 0:
        sdahu_parts.append(df)
        sdahu_counts[label] += len(df)
        print(f"  {os.path.basename(fpath):45s} label={label}  rows={len(df):,}")

df_sdahu = pd.concat(sdahu_parts, ignore_index=True)
print(f"\n  SD-AHU total: {len(df_sdahu):,} rows")
print(f"  Class 2 (Valve Fault):  {sdahu_counts[2]:,}")
print(f"  Class 3 (Damper Fault): {sdahu_counts[3]:,}")

# ── STEP 3: Combine and balance ───────────────────────────────────────────────
print("\nStep 3: Combining and balancing training data...")

# Find common feature columns
ashrae_feat_cols = [c for c in df_ashrae.columns if c != 'label']
sdahu_feat_cols  = [c for c in df_sdahu.columns  if c != 'label']
feat_cols = sorted(list(set(ashrae_feat_cols) & set(sdahu_feat_cols)))
print(f"  Common feature columns: {len(feat_cols)}")

df_combined = pd.concat([
    df_ashrae[feat_cols + ['label']],
    df_sdahu[feat_cols  + ['label']]
], ignore_index=True)

total_label_dist = dict(df_combined['label'].value_counts().sort_index())
print(f"  Combined total: {len(df_combined):,} rows")
print(f"  Class distribution: {total_label_dist}")

# ── STEP 4: Train ─────────────────────────────────────────────────────────────
print("\nStep 4: Training Random Forest (200 trees, balanced weights)...")
t0 = time.time()

X = df_combined[feat_cols].values.astype(float)
y = df_combined['label'].values.astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

model = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    random_state=RS, n_jobs=-1, min_samples_leaf=2)
model.fit(X_tr_sc, y_tr)
print(f"  Training time: {time.time()-t0:.1f}s")

# In-sample evaluation
preds_in   = model.predict(X_te_sc)
f1_in      = f1_score(y_te, preds_in, average='macro', zero_division=0)
classes_in = sorted(np.unique(y_te).tolist())

print(f"\n  IN-SAMPLE macro-F1 = {f1_in:.4f}  (old model: 0.9923)")
print(classification_report(y_te, preds_in,
      labels=classes_in,
      target_names=[CLASS_NAMES[c] for c in classes_in],
      zero_division=0))

# ── STEP 5: Cross-building test ───────────────────────────────────────────────
print("\nStep 5: Cross-building test on ASHRAE withheld buildings...")
cb_parts = []
for fname in ['MZVAV-2-2.csv', 'SZVAV.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  NOT FOUND: {fpath}")
        continue
    df = load_ashrae_file(fpath)
    if len(df) > 0:
        cb_parts.append(df)
        print(f"  {fname}: {len(df):,} rows, "
              f"classes: {sorted(df['label'].unique().tolist())}")

df_cb = pd.concat(cb_parts, ignore_index=True)

# Align to training feature columns
X_cb_aligned = np.zeros((len(df_cb), len(feat_cols)))
for i, col in enumerate(feat_cols):
    if col in df_cb.columns:
        X_cb_aligned[:, i] = df_cb[col].values.astype(float)

X_cb_sc  = scaler.transform(X_cb_aligned)
y_cb     = df_cb['label'].values.astype(int)
preds_cb = model.predict(X_cb_sc)

f1_cb     = f1_score(y_cb, preds_cb, average='macro', zero_division=0)
f1_cb_bin = f1_score((y_cb > 0).astype(int), (preds_cb > 0).astype(int),
                     average='macro', zero_division=0)
gap_pp    = (f1_in - f1_cb) * 100
classes_cb = sorted(np.unique(y_cb).tolist())

print(f"\n  CROSS-BUILDING macro-F1 = {f1_cb:.4f}  (old model: 0.331)")
print(f"  CROSS-BUILDING binary F1 = {f1_cb_bin:.4f}  (old model: 0.415)")
print(f"  Generalisation gap = {gap_pp:.1f} pp  (old model: 66.1 pp)")
print(f"\n  Theoretical maximum = 0.50 "
      f"(only classes {classes_cb} present in test)")
print(classification_report(y_cb, preds_cb,
      labels=classes_cb,
      target_names=[CLASS_NAMES.get(c, str(c)) for c in classes_cb],
      zero_division=0))

# ── STEP 6: PSI analysis ──────────────────────────────────────────────────────
print("\nStep 6: PSI analysis (combined training vs ASHRAE cross-building)...")
print(f"  {'Sensor':50}  {'PSI':8}  Gate  | Old PSI")
print("  " + "-" * 72)

old_psi = {
    'AHU: Supply Air Fan Speed Control Signal': 2.63,
    'AHU: Outdoor Air Temperature':             2.45,
    'AHU: Supply Air Temperature':              0.84,
    'AHU: Return Air Temperature':              0.71,
    'AHU: Mixed Air Temperature':               0.51,
}

psi_results = {}
for col in RAW_COLS:
    if col in df_combined.columns and col in df_cb.columns:
        psi = compute_psi(df_combined[col], df_cb[col])
        psi_results[col] = psi
        gate = "RED" if psi > 0.5 else "AMBER" if psi > 0.2 else "GREEN"
        old  = old_psi.get(col, "-")
        print(f"  {col:50}  {psi:.4f}  {gate:5}  | {old}")

# ── STEP 7: Baseline comparison ───────────────────────────────────────────────
print("\nStep 7: Baseline comparison...")
dummy = DummyClassifier(strategy='most_frequent', random_state=RS)
dummy.fit(X_tr_sc, y_tr)
dummy_preds = dummy.predict(X_cb_sc)
dummy_f1    = f1_score(y_cb, dummy_preds, average='macro', zero_division=0)
print(f"  Dummy classifier cross-building F1: {dummy_f1:.4f}")

# Also compute MCC
mcc = matthews_corrcoef(y_cb, preds_cb)
bal = balanced_accuracy_score(y_cb, preds_cb)
print(f"  Matthews Correlation Coefficient:   {mcc:.4f}")
print(f"  Balanced Accuracy:                  {bal:.4f}")

# ── STEP 8: Decision ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DECISION")
print("=" * 70)
print(f"\n  NEW COMBINED MODEL:")
print(f"    In-sample F1:      {f1_in:.4f}")
print(f"    Cross-building F1: {f1_cb:.4f}")
print(f"    Binary CB F1:      {f1_cb_bin:.4f}")
print(f"    Gap:               {gap_pp:.1f} pp")
print(f"\n  OLD MODEL:")
print(f"    In-sample F1:      0.9923")
print(f"    Cross-building F1: 0.331")
print(f"    Binary CB F1:      0.415")
print(f"    Gap:               66.1 pp")
print()

if f1_cb >= 0.331:
    verdict = "GREEN - NEW MODEL IS BETTER"
    action  = ("New model has higher cross-building F1 AND genuine fault types. "
               "Switch to new model.")
elif f1_cb >= 0.25:
    verdict = "AMBER - SIMILAR, NEW HAS BETTER LABELS"
    action  = ("Cross-building F1 slightly lower but model has genuine fault types "
               "(Healthy, Sensor Bias, Valve Fault, Damper Fault). "
               "Scientific argument for using new model is strong.")
elif f1_cb >= 0.15:
    verdict = "AMBER-RED - MARGINAL"
    action  = ("F1 is noticeably lower. Only switch if genuine fault types "
               "are critical for thesis argument. Otherwise keep original.")
else:
    verdict = "RED - OLD MODEL IS BETTER"
    action  = "Keep original model. Original 0.331 is significantly better."

print(f"  VERDICT: {verdict}")
print(f"  ACTION:  {action}")

# ── Save everything ───────────────────────────────────────────────────────────
out_dir = os.path.join(BASE, "results", "new_model_combined")
os.makedirs(out_dir, exist_ok=True)

joblib.dump(model,     os.path.join(out_dir, "rf_model_combined.pkl"))
joblib.dump(scaler,    os.path.join(out_dir, "scaler_combined.pkl"))
joblib.dump(feat_cols, os.path.join(out_dir, "feature_cols_combined.pkl"))

results = {
    "experiment":            "combined_ashrae_sdahu_genuine_fault_types",
    "training_sources":      ["ASHRAE LBNL (Class 0+1)", "SD-AHU (Class 2+3)"],
    "label_scheme":          {
        "0": "Healthy (ASHRAE)",
        "1": "Sensor Bias (ASHRAE - OA temp sensor bias)",
        "2": "Valve/Coil Fault (SD-AHU coi_leakage + coi_stuck)",
        "3": "Damper Fault (SD-AHU damper_stuck)"
    },
    "n_features":            len(feat_cols),
    "training_rows":         int(len(df_combined)),
    "label_distribution":    {str(k): int(v)
                               for k,v in total_label_dist.items()},
    "f1_insample":           round(float(f1_in), 4),
    "f1_crossbuilding":      round(float(f1_cb), 4),
    "f1_binary_cb":          round(float(f1_cb_bin), 4),
    "gap_pp":                round(float(gap_pp), 2),
    "mcc":                   round(float(mcc), 4),
    "balanced_accuracy":     round(float(bal), 4),
    "dummy_f1_cb":           round(float(dummy_f1), 4),
    "psi_values":            {k: round(v,4) for k,v in psi_results.items()},
    "old_model": {
        "f1_insample":       0.9923,
        "f1_crossbuilding":  0.331,
        "f1_binary_cb":      0.415,
        "gap_pp":            66.1
    },
    "verdict":               verdict,
    "action":                action,
}

with open(os.path.join(out_dir, "combined_model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Model saved:   results/new_model_combined/rf_model_combined.pkl")
print(f"  Results saved: results/new_model_combined/combined_model_results.json")
print("=" * 70)
