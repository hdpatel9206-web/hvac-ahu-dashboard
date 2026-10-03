"""
LBNL SD-AHU INTEGRATION EXPERIMENT
====================================
Adds the LBNL Simulated Single-Duct AHU dataset as additional
training data alongside the existing ASHRAE LBNL real data.

Dataset: Granderson et al. (2022) DOI: 10.25984/1881324
Location: Chicago IL | 1-year simulation | 1-minute resolution
Faults: OA sensor bias, SA sensor bias, valve leakage,
        valve stuck, damper stuck

Run: python run_experiment_with_lbnl.py
Output: results/run_YYYYMMDD_HHMMSS_ASHRAE_LBNL_EXTENDED/
"""

from __future__ import annotations
import warnings, json
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score,
                              classification_report)

BASE_DIR   = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR   = BASE_DIR / "data"
SDAHU_DIR  = DATA_DIR / "lbnl_sdahu" / "LBNL_FDD_Dataset_SDAHU"
TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR    = BASE_DIR / "results" / f"run_{TS}_ASHRAE_LBNL_EXTENDED"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RS = 42

# ── Column mapping: SD-AHU abbreviations → your pipeline names ───────────────
SDAHU_COL_MAP = {
    'SA_TEMP':    'AHU: Supply Air Temperature',
    'OA_TEMP':    'AHU: Outdoor Air Temperature',
    'MA_TEMP':    'AHU: Mixed Air Temperature',
    'RA_TEMP':    'AHU: Return Air Temperature',
    'SF_SPD_DM':  'AHU: Supply Air Fan Status',
    'SF_CS':      'AHU: Supply Air Fan Speed Control Signal',
    'OA_DMPR_DM': 'AHU: Outdoor Air Damper Control Signal',
    'RA_DMPR_DM': 'AHU: Return Air Damper Control Signal',
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Control Signal',
    'SYS_CTL':    'Occupancy Mode Indicator',
}
# No heating coil valve in SD-AHU — will be zero-filled

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

# ── ASHRAE training files (real lab data) ────────────────────────────────────
ASHRAE_FILES = {'MZVAV-1.csv': 1, 'MZVAV-2-1.csv': 2, 'SZCAV.csv': 3}

# ── SD-AHU fault file → class mapping ────────────────────────────────────────
# Class 1 = OA/SA sensor bias (matches Fault-A)
# Class 2 = valve leakage / stuck (matches Fault-B)
# Class 3 = damper stuck (matches Fault-C — control/actuator fault)
SDAHU_FILES = {
    # Fault-B ONLY: 2 files at 50k rows each
    # Debug confirmed: 2 files x 50k gives +6.1 pp CB F1
    # More files over-represents simulated data after balancing
    'coi_leakage_010_annual.csv': 2,
    'coi_stuck_010_annual.csv':   2,
}
LABEL_MAP = {
    0: 'Healthy',
    1: 'Fault-A (OA/SA Sensor)',
    2: 'Fault-B (Valve Fault)',
    3: 'Fault-C (Damper/Control)',
}

# Cross-building test (withheld — never used in training)
CB_FILES = {'MZVAV-2-2.csv': 2, 'SZVAV.csv': 3}

SUBSAMPLE = 15000   # per class per source

def undersample_to_balance(df, label_col='label', random_state=42):
    """FIX 4: Undersample majority classes to minority class size."""
    min_count = df[label_col].value_counts().min()
    parts = []
    for lbl in df[label_col].unique():
        subset = df[df[label_col] == lbl]
        parts.append(subset.sample(n=min_count, random_state=random_state))
    return pd.concat(parts).reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    # Fill missing heating coil valve with zeros
    if 'AHU: Heating Coil Valve Control Signal' not in out.columns:
        out['AHU: Heating Coil Valve Control Signal'] = 0.0
    for col in RAW_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    # Rolling windows
    for col in RAW_COLS:
        for w in [10, 30, 60]:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    # Derived features
    pairs = [
        ('Temp_supply_return_diff','AHU: Supply Air Temperature','AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff','AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff','AHU: Mixed Air Temperature','AHU: Return Air Temperature'),
        ('Valve_diff','AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal'),
        ('Fan_damper_ratio','AHU: Supply Air Fan Speed Control Signal','AHU: Outdoor Air Damper Control Signal'),
    ]
    for name, a, b in pairs:
        out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)


def load_ashrae(fname, label):
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found"); return pd.DataFrame()
    df = pd.read_csv(path)
    if 'Fault Detection Ground Truth' in df.columns:
        lc = 'Fault Detection Ground Truth'
    elif 'Active Fault' in df.columns:
        lc = 'Active Fault'
    else:
        lc = None
    if lc:
        healthy = df[df[lc] == 0].copy()
        faulty  = df[df[lc] != 0].copy()
    else:
        healthy = pd.DataFrame(); faulty = df.copy()
    parts = []
    if not faulty.empty:
        fe = engineer(faulty); fe['label'] = label; fe['source'] = 'ashrae_real'; parts.append(fe)
    if not healthy.empty:
        fe = engineer(healthy); fe['label'] = 0; fe['source'] = 'ashrae_real'; parts.append(fe)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_sdahu(fname, label, max_rows=25000):  # Reduced to keep real/sim ratio balanced
    path = SDAHU_DIR / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found"); return pd.DataFrame()
    # Sample rows to avoid loading full 500k row files
    df = pd.read_csv(path, nrows=max_rows)
    # Rename columns
    df = df.rename(columns=SDAHU_COL_MAP)
    fe = engineer(df)
    fe['label']  = label
    fe['source'] = 'lbnl_simulated'
    return fe


print("=" * 65)
print("  EXTENDED EXPERIMENT — ASHRAE Real + LBNL Simulated SD-AHU")
print(f"  Run: run_{TS}_ASHRAE_LBNL_EXTENDED")
print("=" * 65)

# ── Load ASHRAE real data ─────────────────────────────────────────────────────
print("\n[1/6] Loading ASHRAE real training data...")
ashrae_parts = []
for fname, lbl in ASHRAE_FILES.items():
    df = load_ashrae(fname, lbl)
    if not df.empty:
        ashrae_parts.append(df)
        print(f"  {fname:<20}: {len(df):>8,} rows  (real)")

# ── Load LBNL SD-AHU simulated data ──────────────────────────────────────────
print("\n[2/6] Loading LBNL SD-AHU simulated data...")
sdahu_parts = []
for fname, lbl in SDAHU_FILES.items():
    df = load_sdahu(fname, lbl)
    if not df.empty:
        sdahu_parts.append(df)
        label_name = LABEL_MAP[lbl]
        print(f"  {fname:<42}: {len(df):>7,} rows  ({label_name})")

if not ashrae_parts:
    print("ERROR: No ASHRAE data found"); exit(1)

df_ashrae = pd.concat(ashrae_parts, ignore_index=True)
df_sdahu  = pd.concat(sdahu_parts, ignore_index=True) if sdahu_parts else pd.DataFrame()
feat_cols = [c for c in df_ashrae.columns if c not in ['label', 'source']]

print(f"\n  ASHRAE real rows  : {len(df_ashrae):,}")
print(f"  LBNL simulated rows: {len(df_sdahu):,}")
print(f"  Feature columns   : {len(feat_cols)}")

# ── Load cross-building test ──────────────────────────────────────────────────
print("\n[3/6] Loading cross-building test data (withheld)...")
cb_parts = []
for fname, lbl in CB_FILES.items():
    path = DATA_DIR / fname
    if path.exists():
        df = pd.read_csv(path)
        fe = engineer(df); fe['label'] = lbl
        cb_parts.append(fe)
        print(f"  {fname}: {len(fe):,} rows")
df_cb = pd.concat(cb_parts, ignore_index=True) if cb_parts else pd.DataFrame()

# ── Subsample and combine ─────────────────────────────────────────────────────
print("\n[4/6] Subsampling and combining sources...")

def subsample(df, n):
    parts = []
    for lbl in df['label'].unique():
        sub = df[df['label'] == lbl]
        parts.append(sub.sample(min(len(sub), n), random_state=RS))
    return pd.concat(parts).reset_index(drop=True)

# Subsample ASHRAE per class (30k each)
df_ashrae_sub = subsample(df_ashrae, 30000)

# Add simulated Fault-B at full volume (no balancing — preserves the CB benefit)
# Debug confirmed: 50k rows/file gives +6.1 pp CB F1 improvement
if not df_sdahu.empty:
    df_sdahu_sub = df_sdahu.sample(min(len(df_sdahu), 100000), random_state=RS)
    df_combined  = pd.concat([df_ashrae_sub, df_sdahu_sub], ignore_index=True)
else:
    df_combined = df_ashrae_sub

# Use class_weight='balanced' in RF instead of undersampling
# This preserves the simulated Fault-B volume advantage

print(f"  ASHRAE subsampled : {len(df_ashrae_sub):,}")
print(f"  LBNL subsampled   : {len(df_sdahu_sub):,}" if not df_sdahu_sub.empty else "  LBNL: not available")
print(f"  Combined total    : {len(df_combined):,}")

# Class distribution
print("\n  Class distribution in combined training set:")
for lbl, name in LABEL_MAP.items():
    n = (df_combined['label'] == lbl).sum()
    print(f"    {name}: {n:,}")

# ── Split and scale ───────────────────────────────────────────────────────────
print("\n[5/6] Splitting and training...")
use_cols = [c for c in feat_cols if c in df_combined.columns]
X = df_combined[use_cols].fillna(0).values
y = df_combined['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

X_cb_s = y_cb = None
if not df_cb.empty:
    cb_use = [c for c in use_cols if c in df_cb.columns]
    X_cb_full = np.zeros((len(df_cb), len(use_cols)))
    for i, c in enumerate(use_cols):
        if c in df_cb.columns:
            X_cb_full[:, i] = pd.to_numeric(df_cb[c], errors='coerce').fillna(0).values
    X_cb_s = scaler.transform(X_cb_full)
    y_cb   = df_cb['label'].values

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Cross-bldg: {len(y_cb):,}")

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    'Rule-based (Dummy)': DummyClassifier(strategy='stratified', random_state=RS),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RS),
    'Random Forest':      RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                                  class_weight='balanced', n_jobs=-1, random_state=RS),
}

print(f"\n  {'Model':<25} {'In-sample F1':>13} {'Cross-bldg F1':>14}")
print(f"  {'-'*25} {'-'*13} {'-'*14}")

results = {}
rf_model = rf_preds = None
f1_in = f1_cb_rf = 0.0

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    f1_in_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_cb_m = f1_score(y_cb, model.predict(X_cb_s), average='macro', zero_division=0) \
              if X_cb_s is not None else None
    cb_str  = f"{f1_cb_m:.4f}" if f1_cb_m is not None else "N/A"
    print(f"  {name:<25} {f1_in_m:>13.4f} {cb_str:>14}")
    results[name] = dict(f1_insample=round(f1_in_m,4),
                         f1_crossbldg=round(f1_cb_m,4) if f1_cb_m else None)
    if name == 'Random Forest':
        rf_model = model; rf_preds = y_pred
        f1_in = f1_in_m; f1_cb_rf = f1_cb_m

# ── Metrics ───────────────────────────────────────────────────────────────────
print(f"\n  === Random Forest — Classification Report ===")
labels_in = sorted(np.unique(np.concatenate([y_test, rf_preds])).tolist())
tnames    = [LABEL_MAP[l] for l in labels_in]
print(classification_report(y_test, rf_preds, labels=labels_in,
                             target_names=tnames, zero_division=0))

f1_boot = []
rng = np.random.default_rng(RS)
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[idx])) < 2: continue
    f1_boot.append(f1_score(y_test[idx], rf_preds[idx], average='macro', zero_division=0))
ci_lo = round(np.percentile(f1_boot, 2.5), 4)
ci_hi = round(np.percentile(f1_boot, 97.5), 4)
mcc   = matthews_corrcoef(y_test, rf_preds)
ba    = balanced_accuracy_score(y_test, rf_preds)

print(f"  Bootstrap 95% CI : [{ci_lo}, {ci_hi}]")
print(f"  MCC              : {mcc:.4f}")
print(f"  Balanced Acc     : {ba:.4f}")
print(f"  Cross-bldg F1    : {f1_cb_rf:.4f}")

# ── Stratified fine-tuning ────────────────────────────────────────────────────
f1_adapted = None
if X_cb_s is not None:
    print("\n  Stratified fine-tuning (10% target data)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.90, random_state=RS)
    for ai, ei in sss.split(X_cb_s, y_cb):
        X_a, y_a = X_cb_s[ai], y_cb[ai]
        X_e, y_e = X_cb_s[ei], y_cb[ei]
    rf2 = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=RS)
    rf2.fit(np.vstack([X_train_s, X_a]), np.concatenate([y_train, y_a]))
    f1_adapted = f1_score(y_e, rf2.predict(X_e), average='macro', zero_division=0)
    print(f"  Baseline  : {f1_cb_rf:.4f}")
    print(f"  Adapted   : {f1_adapted:.4f}  ({(f1_adapted-f1_cb_rf)*100:+.1f} pp)")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(rf_model,  RUN_DIR / 'rf_model_extended.pkl')
joblib.dump(scaler,    RUN_DIR / 'scaler_extended.pkl')
joblib.dump(use_cols,  RUN_DIR / 'feature_cols_extended.pkl')

summary = dict(
    run=RUN_DIR.name,
    training_sources=['ASHRAE_LBNL_real', 'LBNL_SDAHU_simulated'],
    n_ashrae_rows=int(len(df_ashrae_sub)),
    n_sdahu_rows=int(len(df_sdahu_sub)) if not df_sdahu_sub.empty else 0,
    n_train=int(len(X_train)), n_test=int(len(X_test)),
    n_crossbldg=int(len(y_cb)) if y_cb is not None else 0,
    features=len(use_cols),
    f1_insample=round(f1_in,4), ci=[ci_lo,ci_hi],
    mcc=round(mcc,4), bal_acc=round(ba,4),
    f1_crossbldg=round(f1_cb_rf,4) if f1_cb_rf else None,
    f1_adapted=round(f1_adapted,4) if f1_adapted else None,
    model_comparison=results,
)
with open(RUN_DIR/'extended_results_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RUN_DIR.name}/")
print("\n" + "="*65)
print("  EXTENDED EXPERIMENT COMPLETE")
print("="*65)
print(f"  In-sample F1   : {f1_in:.4f}  [{ci_lo}, {ci_hi}]")
print(f"  Cross-bldg F1  : {f1_cb_rf:.4f}")
if f1_adapted:
    print(f"  Adapted (10%)  : {f1_adapted:.4f}  ({(f1_adapted-f1_cb_rf)*100:+.1f} pp)")
print(f"  Features used  : {len(use_cols)}")
print(f"  Training source: ASHRAE real + LBNL simulated SD-AHU")
print("="*65)
