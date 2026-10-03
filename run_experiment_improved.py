"""
IMPROVED AHU EXPERIMENT v2 — Ratio Features + Correlation Pruning
==================================================================
Changes from baseline:
  1. Ratio features (scale-invariant, building-agnostic)
  2. Correlation pruning (drops r > 0.95)
  3. Stratified fine-tuning split
  4. Per-building normalisation REMOVED — made cross-bldg worse

Run:  python run_experiment_improved.py
Output: results/run_YYYYMMDD_HHMMSS_ASHRAE_LBNL_IMPROVED/
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

BASE_DIR = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR = BASE_DIR / "data"
TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = BASE_DIR / "results" / f"run_{TS}_ASHRAE_LBNL_IMPROVED"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RS = 42

ROLLING_WINDOWS     = [10, 30, 60]
SUBSAMPLE_PER_CLASS = 30000
CORR_THRESHOLD      = 0.95

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

TRAIN_FILES = {'MZVAV-1.csv': 1, 'MZVAV-2-1.csv': 2, 'SZCAV.csv': 3}
CB_FILES    = {'MZVAV-2-2.csv': 2, 'SZVAV.csv': 3}
LABEL_MAP   = {0:'Healthy', 1:'Fault-A (OA Sensor)', 2:'Fault-B (Valve Leak)', 3:'Fault-C (Control)'}


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
        ('Temp_supply_return_diff',  'AHU: Supply Air Temperature',           'AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff', 'AHU: Outdoor Air Temperature',          'AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff',   'AHU: Mixed Air Temperature',            'AHU: Return Air Temperature'),
        ('Valve_diff',               'AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal'),
        ('Fan_damper_ratio_diff',    'AHU: Supply Air Fan Speed Control Signal','AHU: Outdoor Air Damper Control Signal'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    if 'AHU: Supply Air Temperature' in out.columns and 'AHU: Outdoor Air Temperature' in out.columns:
        out['ratio_supply_outdoor_temp'] = out['AHU: Supply Air Temperature'] / (out['AHU: Outdoor Air Temperature'].abs() + 0.001)
    if 'AHU: Supply Air Fan Speed Control Signal' in out.columns and 'AHU: Outdoor Air Damper Control Signal' in out.columns:
        out['ratio_fan_damper'] = out['AHU: Supply Air Fan Speed Control Signal'] / (out['AHU: Outdoor Air Damper Control Signal'].abs() + 0.001)
    if 'AHU: Cooling Coil Valve Control Signal' in out.columns and 'AHU: Heating Coil Valve Control Signal' in out.columns:
        out['ratio_valve_balance'] = out['AHU: Cooling Coil Valve Control Signal'] / (out['AHU: Heating Coil Valve Control Signal'].abs() + 0.001)
    return out.fillna(0).replace([np.inf, -np.inf], 0)


def load_file(fname, label):
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
        fe = engineer_features(faulty); fe['label'] = label; parts.append(fe)
    if not healthy.empty:
        fe = engineer_features(healthy); fe['label'] = 0; parts.append(fe)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


print("="*65)
print("  IMPROVED AHU EXPERIMENT v2")
print(f"  Run: run_{TS}_ASHRAE_LBNL_IMPROVED")
print("="*65)

print("\n[1/5] Loading training data...")
train_parts = []
for fname, lbl in TRAIN_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        train_parts.append(df)
        print(f"  {fname:<20}: {len(df):>8,} rows")
df_train_full = pd.concat(train_parts, ignore_index=True)
feat_cols = [c for c in df_train_full.columns if c != 'label']
print(f"  Total features before pruning: {len(feat_cols)}")

print("\n[2/5] Pruning highly correlated features (r > 0.95)...")
X_sample = df_train_full[feat_cols].sample(min(5000, len(df_train_full)), random_state=RS).fillna(0)
corr_matrix = X_sample.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
feat_cols_pruned = [f for f in feat_cols if f not in to_drop]
print(f"  Dropped {len(to_drop)} features  |  Final: {len(feat_cols_pruned)}")

print("\n[3/5] Loading cross-building data...")
cb_parts = []
for fname, lbl in CB_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        cb_parts.append(df)
        print(f"  {fname:<20}: {len(df):>8,} rows")
df_cb = pd.concat(cb_parts, ignore_index=True) if cb_parts else pd.DataFrame()

print("\n[4/5] Subsampling, splitting, scaling...")
parts = []
for lbl in df_train_full['label'].unique():
    sub = df_train_full[df_train_full['label'] == lbl]
    parts.append(sub.sample(min(len(sub), SUBSAMPLE_PER_CLASS), random_state=RS))
df_sub = pd.concat(parts).reset_index(drop=True)

X_all = df_sub[feat_cols_pruned].fillna(0).values
y_all = df_sub['label'].values
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=RS)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_cb_s = y_cb = None
if not df_cb.empty:
    X_cb_s = scaler.transform(df_cb[feat_cols_pruned].fillna(0).values)
    y_cb   = df_cb['label'].values
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Cross-bldg: {len(y_cb):,}")

print("\n[5/5] Training models...")
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
f1_insample = f1_cb_rf = 0.0

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    f1_in  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_cb  = f1_score(y_cb, model.predict(X_cb_s), average='macro', zero_division=0) if X_cb_s is not None else None
    print(f"  {name:<25} {f1_in:>13.4f} {str(round(f1_cb,4)) if f1_cb else 'N/A':>14}")
    results[name] = dict(f1_insample=round(f1_in,4), f1_crossbldg=round(f1_cb,4) if f1_cb else None)
    if name == 'Random Forest':
        rf_model = model; rf_preds = y_pred
        f1_insample = f1_in; f1_cb_rf = f1_cb

print(f"\n  === Random Forest — Classification Report ===")
labels_in_test = sorted(np.unique(np.concatenate([y_test, rf_preds])).tolist())
target_names   = [LABEL_MAP[l] for l in labels_in_test]
print(classification_report(y_test, rf_preds, labels=labels_in_test, target_names=target_names, zero_division=0))

f1_boot = []
rng = np.random.default_rng(RS)
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[idx])) < 2: continue
    f1_boot.append(f1_score(y_test[idx], rf_preds[idx], average='macro', zero_division=0))
ci_lo = round(np.percentile(f1_boot, 2.5), 4)
ci_hi = round(np.percentile(f1_boot, 97.5), 4)
mcc = matthews_corrcoef(y_test, rf_preds)
ba  = balanced_accuracy_score(y_test, rf_preds)
print(f"  Bootstrap 95% CI : [{ci_lo}, {ci_hi}]")
print(f"  MCC              : {mcc:.4f}")
print(f"  Balanced Acc     : {ba:.4f}")
print(f"  Cross-bldg F1    : {f1_cb_rf:.4f}")

f1_adapted = None
if X_cb_s is not None:
    print("\n  Stratified fine-tuning (10%)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.90, random_state=RS)
    for ai, ei in sss.split(X_cb_s, y_cb):
        X_a, y_a = X_cb_s[ai], y_cb[ai]
        X_e, y_e = X_cb_s[ei], y_cb[ei]
    rf2 = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=RS)
    rf2.fit(np.vstack([X_train_s, X_a]), np.concatenate([y_train, y_a]))
    f1_adapted = f1_score(y_e, rf2.predict(X_e), average='macro', zero_division=0)
    print(f"  Baseline: {f1_cb_rf:.4f}  →  Adapted: {f1_adapted:.4f}  ({(f1_adapted-f1_cb_rf)*100:+.1f} pp)")

joblib.dump(rf_model,         RUN_DIR / 'rf_model_improved.pkl')
joblib.dump(scaler,           RUN_DIR / 'scaler_improved.pkl')
joblib.dump(feat_cols_pruned, RUN_DIR / 'feature_cols_improved.pkl')

summary = dict(run=RUN_DIR.name, improvements=['ratio_features','correlation_pruning','stratified_finetuning'],
               features_before=len(feat_cols), features_dropped=len(to_drop), features_final=len(feat_cols_pruned),
               n_train=int(len(X_train)), n_test=int(len(X_test)), n_crossbldg=int(len(y_cb)) if y_cb is not None else 0,
               f1_insample=round(f1_insample,4), ci=[ci_lo,ci_hi], mcc=round(mcc,4), bal_acc=round(ba,4),
               f1_crossbldg=round(f1_cb_rf,4) if f1_cb_rf else None,
               f1_adapted=round(f1_adapted,4) if f1_adapted else None,
               model_comparison=results)
with open(RUN_DIR / 'improved_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RUN_DIR.name}/")
print("\n"+"="*65)
print("  COMPLETE")
print("="*65)
print(f"  In-sample F1  : {f1_insample:.4f}  [{ci_lo}, {ci_hi}]")
print(f"  Cross-bldg F1 : {f1_cb_rf:.4f}")
if f1_adapted: print(f"  Adapted (10%) : {f1_adapted:.4f}  ({(f1_adapted-f1_cb_rf)*100:+.1f} pp)")
print(f"  Features used : {len(feat_cols_pruned)}")
print("="*65)
