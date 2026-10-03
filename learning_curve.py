"""
LEARNING CURVE EXPERIMENT — Fixed Version
==========================================
Fixes applied:
  FIX 1: Rolling windows computed per-file inside load_file()
          before concatenation — no cross-file boundary leakage
  FIX 2: Healthy rows ONLY from Active Fault==0 rows
          No fabricated healthy data from fault-only files
  FIX 3: Stratified subsampling at each training proportion

Run: python learning_curve.py
"""

from __future__ import annotations
import warnings, json
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR = BASE_DIR / "data"
RS = 42
ROLLING_WINDOWS = [10, 30, 60]
SUBSAMPLE_PER_CLASS = 30000

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
TRAIN_FILES = {'MZVAV-1.csv':1, 'MZVAV-2-1.csv':2, 'SZCAV.csv':3}
CB_FILES    = {'MZVAV-2-2.csv':2, 'SZVAV.csv':3}
LABEL_MAP   = {0:'Healthy',1:'Fault-A',2:'Fault-B',3:'Fault-C'}

def engineer_features(df):
    """FIX 1: Called on single file — no cross-file rolling leakage."""
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in available:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    pairs = [
        ('Temp_supply_return_diff','AHU: Supply Air Temperature','AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff','AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff','AHU: Mixed Air Temperature','AHU: Return Air Temperature'),
        ('Valve_diff','AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal'),
        ('Fan_damper_ratio','AHU: Supply Air Fan Speed Control Signal','AHU: Outdoor Air Damper Control Signal'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)

def load_file(fname, label):
    """
    FIX 1: engineer_features called on single file before concat.
    FIX 2: healthy rows ONLY from Active Fault==0.
            No healthy data fabricated from fault-only files.
    """
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found"); return pd.DataFrame()
    df = pd.read_csv(path)
    # Detect label column — ASHRAE uses 'Fault Detection Ground Truth'
    # Value 0 = fault-free (healthy), Value 1 = fault active
    if 'Fault Detection Ground Truth' in df.columns:
        label_col   = 'Fault Detection Ground Truth'
    elif 'Active Fault' in df.columns:
        label_col   = 'Active Fault'
    else:
        label_col   = None

    if label_col:
        healthy_raw = df[df[label_col]==0].copy()
        faulty_raw  = df[df[label_col]!=0].copy()
    else:
        healthy_raw = pd.DataFrame()
        faulty_raw  = df.copy()
    parts = []
    if not faulty_raw.empty:
        fe = engineer_features(faulty_raw); fe['label'] = label; parts.append(fe)
    if not healthy_raw.empty:
        fe = engineer_features(healthy_raw); fe['label'] = 0; parts.append(fe)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

print("="*60)
print("  LEARNING CURVE — Fixed Version")
print("="*60)

print("\n[1/4] Loading training data...")
train_parts = []
for fname, lbl in TRAIN_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        train_parts.append(df)
        h = (df['label']==0).sum(); f = (df['label']!=0).sum()
        print(f"  {fname:<20}: {len(df):>8,} (fault={f:,}, healthy={h:,})")

df_all    = pd.concat(train_parts, ignore_index=True)
feat_cols = [c for c in df_all.columns if c != 'label']
print(f"\n  Total: {len(df_all):,} rows | {len(feat_cols)} features")
print("  Classes:", {LABEL_MAP[l]: int((df_all['label']==l).sum())
                     for l in df_all['label'].unique()})

print("\n[2/4] Loading cross-building data...")
cb_parts = []
for fname, lbl in CB_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        cb_parts.append(df); print(f"  {fname}: {len(df):,} rows")
df_cb = pd.concat(cb_parts, ignore_index=True) if cb_parts else pd.DataFrame()

print("\n[3/4] Building train/test split...")
parts = []
for lbl in df_all['label'].unique():
    sub = df_all[df_all['label']==lbl]
    parts.append(sub.sample(min(len(sub), SUBSAMPLE_PER_CLASS), random_state=RS))
df_sub = pd.concat(parts).reset_index(drop=True)

X_all = df_sub[feat_cols].fillna(0).values
y_all = df_sub['label'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RS)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

X_cb_s = y_cb = None
if not df_cb.empty:
    X_cb_s = sc.transform(df_cb[feat_cols].fillna(0).values)
    y_cb   = df_cb['label'].values

print(f"  Train: {len(X_tr):,} | Test: {len(X_te):,} | CB: {len(y_cb):,}")

print("\n[4/4] Running learning curve...")
print(f"\n  {'Pct':>5} {'N':>8} {'In-samp F1':>12} {'Cross-bldg F1':>14}")
print(f"  {'-'*5} {'-'*8} {'-'*12} {'-'*14}")

results = []
for pct in [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]:
    idx = []
    for lbl in np.unique(y_tr):
        li = np.where(y_tr==lbl)[0]
        idx.extend(np.random.RandomState(RS).choice(li, max(1,int(len(li)*pct)), replace=False))
    idx = np.array(idx)
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 class_weight='balanced', n_jobs=-1, random_state=RS)
    rf.fit(X_tr_s[idx], y_tr[idx])
    f1_in = f1_score(y_te, rf.predict(X_te_s), average='macro', zero_division=0)
    f1_cb = f1_score(y_cb, rf.predict(X_cb_s), average='macro', zero_division=0) \
            if X_cb_s is not None else None
    print(f"  {pct*100:>4.0f}% {len(idx):>8,} {f1_in:>12.4f} "
          f"{str(round(f1_cb,4)) if f1_cb else 'N/A':>14}")
    results.append({'pct':pct,'n':int(len(idx)),'f1_in':round(f1_in,4),
                    'f1_cb':round(f1_cb,4) if f1_cb else None})

out = BASE_DIR/'results'
out.mkdir(exist_ok=True)
with open(out/'learning_curve_results.json','w') as f:
    json.dump(results, f, indent=2)
pd.DataFrame(results).to_csv(out/'learning_curve_results.csv', index=False)
print("\nSaved to results/learning_curve_results.json")
print("="*60)
