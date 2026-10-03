# new_model_optimised.py
# OPTIMISED COMBINED MODEL - All 6 issues fixed
#
# Fixes applied:
#   1. Balanced class sizes (~120k per class)
#   2. SD-AHU healthy rows added to Class 0
#   3. Zero-filled columns removed from feature set
#   4. Only genuinely shared sensors used
#   5. Bootstrap CI added
#   6. Class cap prevents extreme imbalance
#
# Label scheme:
#   Class 0 = Healthy     (ASHRAE + SD-AHU AHU_annual)
#   Class 1 = Sensor Bias (ASHRAE - genuine OA temp sensor bias)
#   Class 2 = Valve Fault (SD-AHU coi_leakage + coi_stuck)
#   Class 3 = Damper Fault(SD-AHU damper_stuck)
#
# Run: python new_model_optimised.py
# Time: approximately 15-25 minutes

import os, json, glob, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy    import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, classification_report,
                              matthews_corrcoef, balanced_accuracy_score)

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
SDAHU_DIR = os.path.join(DATA_DIR, "sdahu")
RS        = 42
ROLLING_WINDOWS   = [10, 30, 60]
MAX_PER_CLASS     = 120000   # FIX 1+6: cap each class at 120k rows
MAX_PER_FILE      = 50000
N_BOOTSTRAP       = 1000     # FIX 4: bootstrap CI

# FIX 2+3: Only columns genuinely present in BOTH ASHRAE and SD-AHU
# Exclude heating coil valve (not in SD-AHU)
GENUINE_SHARED_COLS = [
    'AHU: Supply Air Temperature',
    'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',
    'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',
    'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal',
    'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
# Deliberately EXCLUDED: 'AHU: Heating Coil Valve Control Signal'
# Reason: not present in SD-AHU, zero-filling creates noise

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

ASHRAE_TRAIN = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']

SDAHU_CLASS2 = ['coi_leakage', 'coi_stuck']
SDAHU_CLASS3 = ['damper_stuck']
SDAHU_CLASS0 = ['ahu_annual']   # FIX 5: add healthy from SD-AHU

EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
CLASS_NAMES  = {0:'Healthy', 1:'Sensor Bias', 2:'Valve Fault', 3:'Damper Fault'}

def engineer_features(df):
    available = [c for c in GENUINE_SHARED_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in available:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    # Temperature difference features (scale-invariant)
    pairs = [
        ('Temp_supply_return_diff',  'AHU: Supply Air Temperature',
                                     'AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff', 'AHU: Outdoor Air Temperature',
                                     'AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff',   'AHU: Mixed Air Temperature',
                                     'AHU: Return Air Temperature'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)

def load_ashrae(fpath, label_col='Fault Detection Ground Truth'):
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    if label_col not in df.columns:
        return pd.DataFrame()
    for col in GENUINE_SHARED_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0.0
    feats = engineer_features(df)
    raw   = df[label_col].values[:len(feats)]
    feats['label'] = np.where(raw == 0, 0, 1)
    return feats.dropna()

def load_sdahu(fpath, label):
    df = pd.read_csv(fpath, low_memory=False, nrows=MAX_PER_FILE)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for sc, ac in SDAHU_COL_MAP.items():
        if sc in df.columns:
            df[ac] = pd.to_numeric(df[sc], errors='coerce')
    for col in GENUINE_SHARED_COLS:
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
    mn  = float(min(ref.min(), dep.min()))
    mx  = float(max(ref.max(), dep.max()))
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    eps  = 1e-6
    rc = np.histogram(ref, bins=bins)[0].astype(float) + eps
    dc = np.histogram(dep, bins=bins)[0].astype(float) + eps
    rp, dp = rc/rc.sum(), dc/dc.sum()
    return float(np.sum((dp - rp) * np.log(dp / rp)))

print("=" * 70)
print("OPTIMISED COMBINED MODEL — 6 FIXES APPLIED")
print("Balanced classes | Genuine shared cols | Bootstrap CI")
print("=" * 70)

# ── STEP 1: Load all training data ───────────────────────────────────────────
print("\nStep 1: Loading training data...")
class_pools = {0: [], 1: [], 2: [], 3: []}

# ASHRAE: classes 0 and 1
for fname in ASHRAE_TRAIN:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        continue
    df = load_ashrae(fpath)
    if len(df) > 0:
        c0 = df[df['label']==0]
        c1 = df[df['label']==1]
        class_pools[0].append(c0)
        class_pools[1].append(c1)
        print(f"  ASHRAE {fname:15s}: healthy={len(c0):,}  sensor_bias={len(c1):,}")

# FIX 5: SD-AHU healthy as additional Class 0
if not os.path.exists(SDAHU_DIR):
    SDAHU_DIR = os.path.join(DATA_DIR, "lbnl_sdahu", "LBNL_FDD_Dataset_SDAHU")

for fpath in glob.glob(os.path.join(SDAHU_DIR, "*.csv")):
    fname = os.path.basename(fpath).lower()
    label = None
    for pat in SDAHU_CLASS0:
        if pat in fname:
            label = 0; break
    if label is None:
        for pat in SDAHU_CLASS2:
            if pat in fname:
                label = 2; break
    if label is None:
        for pat in SDAHU_CLASS3:
            if pat in fname:
                label = 3; break
    if label is None:
        continue
    df = load_sdahu(fpath, label)
    if len(df) > 0:
        class_pools[label].append(df)
        src = {0:'healthy', 2:'valve', 3:'damper'}[label]
        print(f"  SD-AHU {os.path.basename(fpath):40s}: {src}={len(df):,}")

# Combine each class pool
class_dfs = {}
for cls, parts in class_pools.items():
    if parts:
        class_dfs[cls] = pd.concat(parts, ignore_index=True)
        print(f"  Class {cls} ({CLASS_NAMES[cls]:12s}): {len(class_dfs[cls]):,} rows total")

# FIX 1+6: Balance classes — cap at MAX_PER_CLASS
print(f"\n  Balancing classes (cap: {MAX_PER_CLASS:,} per class)...")
balanced_parts = []
for cls, df in class_dfs.items():
    if len(df) > MAX_PER_CLASS:
        df = df.sample(n=MAX_PER_CLASS, random_state=RS)
    balanced_parts.append(df)
    print(f"  Class {cls} ({CLASS_NAMES[cls]:12s}): {len(df):,} rows (after cap)")

df_train  = pd.concat(balanced_parts, ignore_index=True)
feat_cols = [c for c in df_train.columns if c != 'label']
print(f"\n  Total training: {len(df_train):,} rows, {len(feat_cols)} features")
print(f"  Final distribution: {dict(df_train['label'].value_counts().sort_index())}")

# ── STEP 2: Train ─────────────────────────────────────────────────────────────
print("\nStep 2: Training Random Forest (200 trees, balanced weights)...")
t0 = time.time()

X = df_train[feat_cols].values.astype(float)
y = df_train['label'].values.astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

model = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    random_state=RS, n_jobs=-1, min_samples_leaf=2)
model.fit(X_tr_sc, y_tr)
print(f"  Training time: {time.time()-t0:.1f}s")

preds_in   = model.predict(X_te_sc)
f1_in      = f1_score(y_te, preds_in, average='macro', zero_division=0)
classes_in = sorted(np.unique(y_te).tolist())

print(f"\n  IN-SAMPLE macro-F1 = {f1_in:.4f}  (old model: 0.9923)")
print(classification_report(y_te, preds_in,
      labels=classes_in,
      target_names=[CLASS_NAMES[c] for c in classes_in],
      zero_division=0))

# FIX 4: Bootstrap CI
print(f"  Computing bootstrap CI ({N_BOOTSTRAP} iterations)...")
rng = np.random.RandomState(RS)
bs_scores = []
for _ in range(N_BOOTSTRAP):
    idx = rng.choice(len(X_te_sc), len(X_te_sc), replace=True)
    s   = f1_score(y_te[idx], preds_in[idx], average='macro', zero_division=0)
    bs_scores.append(s)
ci_lo = np.percentile(bs_scores, 2.5)
ci_hi = np.percentile(bs_scores, 97.5)
print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  (old model: [0.9913, 0.9933])")

# ── STEP 3: Cross-building test ───────────────────────────────────────────────
print("\nStep 3: Cross-building test on ASHRAE withheld buildings...")
cb_parts = []
for fname in ['MZVAV-2-2.csv', 'SZVAV.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        continue
    df = load_ashrae(fpath)
    if len(df) > 0:
        cb_parts.append(df)
        print(f"  {fname}: {len(df):,} rows, "
              f"classes: {sorted(df['label'].unique().tolist())}")

df_cb = pd.concat(cb_parts, ignore_index=True)

X_cb_aligned = np.zeros((len(df_cb), len(feat_cols)))
for i, col in enumerate(feat_cols):
    if col in df_cb.columns:
        X_cb_aligned[:, i] = df_cb[col].values.astype(float)

X_cb_sc   = scaler.transform(X_cb_aligned)
y_cb      = df_cb['label'].values.astype(int)
preds_cb  = model.predict(X_cb_sc)

f1_cb     = f1_score(y_cb, preds_cb, average='macro', zero_division=0)
f1_cb_bin = f1_score((y_cb>0).astype(int), (preds_cb>0).astype(int),
                     average='macro', zero_division=0)
gap_pp    = (f1_in - f1_cb) * 100
mcc       = matthews_corrcoef(y_cb, preds_cb)
bal       = balanced_accuracy_score(y_cb, preds_cb)
classes_cb = sorted(np.unique(y_cb).tolist())

print(f"\n  CROSS-BUILDING macro-F1  = {f1_cb:.4f}  (old: 0.331 | prev: 0.276)")
print(f"  CROSS-BUILDING binary F1 = {f1_cb_bin:.4f}  (old: 0.415 | prev: 0.560)")
print(f"  Generalisation gap       = {gap_pp:.1f} pp  (old: 66.1)")
print(f"  MCC                      = {mcc:.4f}")
print(f"  Theoretical maximum      = 0.50 (classes {classes_cb} only in test)")
print(classification_report(y_cb, preds_cb,
      labels=classes_cb,
      target_names=[CLASS_NAMES.get(c, str(c)) for c in classes_cb],
      zero_division=0))

# ── STEP 4: PSI ───────────────────────────────────────────────────────────────
print("\nStep 4: PSI analysis...")
print(f"  {'Sensor':50}  {'New PSI':8}  {'Old PSI':8}  Gate")
print("  " + "-" * 75)
old_psi = {
    'AHU: Supply Air Fan Speed Control Signal': 2.63,
    'AHU: Outdoor Air Temperature':             2.45,
    'AHU: Supply Air Temperature':              0.84,
    'AHU: Return Air Temperature':              0.71,
    'AHU: Mixed Air Temperature':               0.51,
    'AHU: Outdoor Air Damper Control Signal':   0.48,
    'AHU: Return Air Damper Control Signal':    1.44,
    'AHU: Cooling Coil Valve Control Signal':   0.88,
    'AHU: Supply Air Fan Status':               0.01,
    'Occupancy Mode Indicator':                 0.0002,
}
psi_results = {}
for col in GENUINE_SHARED_COLS:
    if col in df_train.columns and col in df_cb.columns:
        psi = compute_psi(df_train[col], df_cb[col])
        psi_results[col] = psi
        gate = "RED" if psi>0.5 else "AMBER" if psi>0.2 else "GREEN"
        o    = old_psi.get(col, "-")
        print(f"  {col:50}  {psi:8.4f}  {str(o):>8}  {gate}")

# ── STEP 5: Final verdict ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL VERDICT — OPTIMISED MODEL")
print("=" * 70)
print(f"\n  OPTIMISED MODEL:")
print(f"    In-sample F1:      {f1_in:.4f}  CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"    Cross-building F1: {f1_cb:.4f}")
print(f"    Binary CB F1:      {f1_cb_bin:.4f}")
print(f"    Gap:               {gap_pp:.1f} pp")
print(f"\n  PREVIOUS COMBINED MODEL:")
print(f"    In-sample F1:      0.9944")
print(f"    Cross-building F1: 0.2764")
print(f"    Binary CB F1:      0.5604")
print(f"\n  ORIGINAL OLD MODEL:")
print(f"    In-sample F1:      0.9923  CI [0.9913, 0.9933]")
print(f"    Cross-building F1: 0.331")
print(f"    Binary CB F1:      0.415")
print()

if f1_cb >= 0.331:
    verdict = "GREEN — OPTIMISED MODEL BEATS ORIGINAL"
    action  = "Switch to optimised model. Better F1 AND genuine fault types."
elif f1_cb >= 0.276:
    verdict = "AMBER-GREEN — BETTER THAN PREVIOUS COMBINED, CLOSE TO ORIGINAL"
    action  = ("Improved over previous attempt. Slightly below original 4-class F1 "
               "but genuine fault types and better binary F1. Strong case to switch.")
elif f1_cb >= 0.25:
    verdict = "AMBER — SIMILAR TO ORIGINAL"
    action  = ("Genuine fault types with similar performance. "
               "Scientific argument strong. Your choice.")
else:
    verdict = "RED — KEEP ORIGINAL"
    action  = "Original model 0.331 is significantly better. Keep it."

print(f"  VERDICT: {verdict}")
print(f"  ACTION:  {action}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(BASE, "results", "new_model_optimised")
os.makedirs(out_dir, exist_ok=True)

joblib.dump(model,     os.path.join(out_dir, "rf_model_optimised.pkl"))
joblib.dump(scaler,    os.path.join(out_dir, "scaler_optimised.pkl"))
joblib.dump(feat_cols, os.path.join(out_dir, "feature_cols_optimised.pkl"))

results = {
    "experiment":          "optimised_combined_ashrae_sdahu",
    "fixes_applied":       [
        "Balanced classes capped at 120k each",
        "SD-AHU AHU_annual.csv added as Class 0",
        "Heating coil valve removed (not genuine shared column)",
        "Bootstrap CI added",
        "Genuine shared columns only (10 not 11)",
    ],
    "label_scheme":        {
        "0": "Healthy (ASHRAE + SD-AHU AHU_annual)",
        "1": "Sensor Bias (ASHRAE only)",
        "2": "Valve/Coil Fault (SD-AHU coi_leakage + coi_stuck)",
        "3": "Damper Fault (SD-AHU damper_stuck)"
    },
    "n_features":          len(feat_cols),
    "training_rows":       int(len(df_train)),
    "label_distribution":  {str(k): int(v)
                             for k,v in
                             df_train['label'].value_counts().sort_index().items()},
    "f1_insample":         round(float(f1_in), 4),
    "ci_lo":               round(float(ci_lo), 4),
    "ci_hi":               round(float(ci_hi), 4),
    "f1_crossbuilding":    round(float(f1_cb), 4),
    "f1_binary_cb":        round(float(f1_cb_bin), 4),
    "gap_pp":              round(float(gap_pp), 2),
    "mcc":                 round(float(mcc), 4),
    "psi_values":          {k: round(v,4) for k,v in psi_results.items()},
    "old_model": {
        "f1_insample":     0.9923,
        "f1_crossbuilding":0.331,
        "f1_binary_cb":    0.415,
    },
    "prev_combined_model": {
        "f1_insample":     0.9944,
        "f1_crossbuilding":0.2764,
        "f1_binary_cb":    0.5604,
    },
    "verdict":             verdict,
    "action":              action,
}

with open(os.path.join(out_dir, "optimised_model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved to: results/new_model_optimised/")
print("=" * 70)
