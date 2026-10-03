# new_model_training.py - FIXED VERSION
# Fixes: AHU_annual.csv loaded as Class 0, classification_report labels fixed
# Run: python new_model_training.py

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

# FIXED: AHU_annual.csv explicitly mapped to Class 0
SDAHU_LABEL_MAP = {
    'ahu_annual':  0,   # Healthy - FIXED
    'oa_bias':     1,   # Sensor bias
    'coi_bias':    1,   # Sensor bias
    'coi_leakage': 2,   # Valve fault
    'coi_stuck':   2,   # Valve fault
    'damper_stuck':3,   # Damper fault
}

EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

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

def load_sdahu_file(fpath, label):
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

def load_ashrae_file(fpath):
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
    feats['label'] = df[lc].values[:len(feats)]
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

print("=" * 65)
print("NEW MODEL TRAINING - FIXED VERSION")
print("Classes: 0=Healthy  1=Sensor Bias  2=Valve Fault  3=Damper")
print("NOTE: Model already trained (10 min). Skipping to evaluation.")
print("=" * 65)

# ── STEP 1: Load SD-AHU ───────────────────────────────────────────────────────
print("\nStep 1: Loading SD-AHU training data (including AHU_annual.csv)...")

if not os.path.exists(SDAHU_DIR):
    SDAHU_DIR = os.path.join(DATA_DIR, "lbnl_sdahu", "LBNL_FDD_Dataset_SDAHU")

all_sdahu = glob.glob(os.path.join(SDAHU_DIR, "*.csv"))
train_parts = []
label_counts = {0:0, 1:0, 2:0, 3:0}

for fpath in sorted(all_sdahu):
    fname = os.path.basename(fpath).lower()
    label = None
    for key, lbl in SDAHU_LABEL_MAP.items():
        if key in fname:
            label = lbl
            break
    if label is None:
        print(f"  SKIP: {os.path.basename(fpath)}")
        continue
    df = load_sdahu_file(fpath, label)
    if len(df) > 0:
        train_parts.append(df)
        label_counts[label] += len(df)
        print(f"  {os.path.basename(fpath):45s} label={label}  rows={len(df):,}")

df_train = pd.concat(train_parts, ignore_index=True)
feat_cols = [c for c in df_train.columns if c != 'label']
print(f"\n  Total: {len(df_train):,} rows, {len(feat_cols)} features")
print(f"  Class distribution: {label_counts}")

# ── STEP 2: Train ─────────────────────────────────────────────────────────────
print("\nStep 2: Training Random Forest...")
t0 = time.time()

X = df_train[feat_cols].values.astype(float)
y = df_train['label'].values.astype(int)

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

preds_in = model.predict(X_te_sc)
f1_in = f1_score(y_te, preds_in, average='macro', zero_division=0)
classes_in = sorted(np.unique(y_te).tolist())
names_in = {0:'Healthy', 1:'Sensor Bias', 2:'Valve Fault', 3:'Damper Fault'}

print(f"\n  IN-SAMPLE macro-F1 = {f1_in:.4f}  (old model: 0.9923)")
print(classification_report(y_te, preds_in,
      labels=classes_in,
      target_names=[names_in[c] for c in classes_in],
      zero_division=0))

# ── STEP 3: Cross-building ────────────────────────────────────────────────────
print("\nStep 3: Cross-building test on ASHRAE withheld buildings...")
cb_parts = []
for fname in ['MZVAV-2-2.csv', 'SZVAV.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        continue
    df = load_ashrae_file(fpath)
    if len(df) > 0:
        cb_parts.append(df)
        print(f"  {fname}: {len(df):,} rows, classes: {sorted(df['label'].unique().tolist())}")

df_cb = pd.concat(cb_parts, ignore_index=True)

# Align columns
X_cb_aligned = np.zeros((len(df_cb), len(feat_cols)))
for i, col in enumerate(feat_cols):
    if col in df_cb.columns:
        X_cb_aligned[:, i] = df_cb[col].values.astype(float)

X_cb_sc  = scaler.transform(X_cb_aligned)
y_cb     = df_cb['label'].values.astype(int)
preds_cb = model.predict(X_cb_sc)
f1_cb    = f1_score(y_cb, preds_cb, average='macro', zero_division=0)
f1_cb_bin = f1_score((y_cb > 0).astype(int), (preds_cb > 0).astype(int),
                      average='macro', zero_division=0)

classes_cb = sorted(np.unique(y_cb).tolist())
print(f"\n  CROSS-BUILDING macro-F1 = {f1_cb:.4f}  (old model: 0.331)")
print(f"  CROSS-BUILDING binary F1 = {f1_cb_bin:.4f}  (old model: 0.415)")
print(f"  Gap from in-sample = {(f1_in - f1_cb)*100:.1f} pp")
print(classification_report(y_cb, preds_cb,
      labels=classes_cb,
      target_names=[names_in.get(c, str(c)) for c in classes_cb],
      zero_division=0))

# ── STEP 4: PSI ───────────────────────────────────────────────────────────────
print("\nStep 4: PSI (SD-AHU training vs ASHRAE cross-building)...")
print(f"  {'Sensor':50}  {'PSI':8}  Gate")
print("  " + "-" * 62)
psi_results = {}
for col in RAW_COLS:
    if col in df_train.columns and col in df_cb.columns:
        psi = compute_psi(df_train[col], df_cb[col])
        psi_results[col] = psi
        gate = "RED" if psi > 0.5 else "AMBER" if psi > 0.2 else "GREEN"
        print(f"  {col:50}  {psi:.4f}  {gate}")

# ── STEP 5: Decision ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL VERDICT")
print("=" * 65)
print(f"\n  NEW MODEL:  in-sample={f1_in:.4f}  cross-building={f1_cb:.4f}")
print(f"  OLD MODEL:  in-sample=0.9923       cross-building=0.331")
print()

if f1_cb >= 0.331:
    verdict = "GREEN - NEW MODEL IS BETTER"
    action  = "Switch to new model. Genuine fault types. Update chapters."
elif f1_cb >= 0.28:
    verdict = "AMBER - SIMILAR PERFORMANCE"
    action  = "New model has genuine fault types. Your choice to switch or keep old."
else:
    verdict = "RED - OLD MODEL IS BETTER"
    action  = "Keep original model. Restore from backup if needed."

print(f"  VERDICT: {verdict}")
print(f"  ACTION:  {action}")

# Save
out_dir = os.path.join(BASE, "results", "new_model_sdahu")
os.makedirs(out_dir, exist_ok=True)
joblib.dump(model,     os.path.join(out_dir, "rf_model_new.pkl"))
joblib.dump(scaler,    os.path.join(out_dir, "scaler_new.pkl"))
joblib.dump(feat_cols, os.path.join(out_dir, "feature_cols_new.pkl"))

results = {
    "experiment":        "new_model_sdahu_genuine_fault_types",
    "training_data":     "LBNL SD-AHU with genuine fault type labels",
    "label_scheme":      "0=Healthy 1=SensorBias 2=ValveFault 3=DamperFault",
    "f1_insample":       round(float(f1_in), 4),
    "f1_crossbuilding":  round(float(f1_cb), 4),
    "f1_binary_cb":      round(float(f1_cb_bin), 4),
    "gap_pp":            round((f1_in - f1_cb)*100, 2),
    "old_f1_insample":   0.9923,
    "old_f1_crossbldg":  0.331,
    "psi_values":        {k: round(v,4) for k,v in psi_results.items()},
    "label_distribution":{str(k): v for k,v in label_counts.items()},
    "verdict":           verdict,
    "action":            action,
}
with open(os.path.join(out_dir, "new_model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved to: results/new_model_sdahu/new_model_results.json")
print("=" * 65)
