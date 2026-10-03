"""
PROPERLY FIXED FINE-TUNING EVALUATION
=======================================
Cross-building test contains ONLY Class 2 and Class 3.
Theoretical max macro-F1 (4-class) = 0.50.

This script reports BOTH:
1. Macro-F1 (4-class) — comparable to original results
2. Macro-F1 (2-class, present only) — fair metric

Run: python pipeline/finetune_fixed.py
"""
import warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RS = 42

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
LABEL_MAP = {0:'Healthy',1:'Fault-A',2:'Fault-B',3:'Fault-C'}
TRAIN_FILES = {'MZVAV-1.csv':1,'MZVAV-2-1.csv':2,'SZCAV.csv':3}
CB_FILES    = {'MZVAV-2-2.csv':2,'SZVAV.csv':3}

def engineer(df):
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in available:
        for w in [10,30,60]:
            out[f'{col}_rm{w}'] = out[col].rolling(w,min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w,min_periods=1).std().fillna(0)
    pairs = [
        ('Temp_supply_return_diff','AHU: Supply Air Temperature','AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff','AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff','AHU: Mixed Air Temperature','AHU: Return Air Temperature'),
        ('Valve_diff','AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal'),
        ('Fan_damper_ratio','AHU: Supply Air Fan Speed Control Signal','AHU: Outdoor Air Damper Control Signal'),
    ]
    for name,a,b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a]-out[b]
    return out.fillna(0).replace([np.inf,-np.inf],0)

def load_file(fname, label):
    path = DATA_DIR / fname
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    # FIX 2: healthy ONLY from ground truth label column
    # ASHRAE files use 'Fault Detection Ground Truth' (0=healthy, 1=fault)
    if 'Fault Detection Ground Truth' in df.columns:
        lc = 'Fault Detection Ground Truth'
    elif 'Active Fault' in df.columns:
        lc = 'Active Fault'
    else:
        lc = None
    if lc:
        h = df[df[lc]==0].copy()
        f = df[df[lc]!=0].copy()
    else:
        h = pd.DataFrame(); f = df.copy()
    parts = []
    if not f.empty:
        fe = engineer(f); fe['label'] = label; parts.append(fe)
    if not h.empty:
        fe = engineer(h); fe['label'] = 0; parts.append(fe)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def to_X(df, feat_cols, scaler):
    X = np.zeros((len(df), len(feat_cols)))
    for i,c in enumerate(feat_cols):
        if c in df.columns:
            X[:,i] = pd.to_numeric(df[c],errors='coerce').fillna(0).values
    return scaler.transform(X)

print("="*65)
print("  PROPERLY FIXED FINE-TUNING EVALUATION")
print("="*65)

model     = joblib.load(BASE_DIR/'models'/'rf_model.pkl')
feat_cols = joblib.load(BASE_DIR/'models'/'feature_cols.pkl')
scaler    = joblib.load(BASE_DIR/'models'/'scaler.pkl')

# Load training data
print("\nLoading training data...")
parts = []
for fname,lbl in TRAIN_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty: parts.append(df)
df_train = pd.concat(parts, ignore_index=True)
subs = []
for lbl in df_train['label'].unique():
    s = df_train[df_train['label']==lbl]
    subs.append(s.sample(min(len(s),30000),random_state=RS))
df_sub    = pd.concat(subs).reset_index(drop=True)
avail     = [c for c in feat_cols if c in df_sub.columns]
X_tr_raw  = df_sub[avail].fillna(0).values
y_train   = df_sub['label'].values
X_tr_full = np.zeros((len(X_tr_raw), len(feat_cols)))
X_tr_full[:,:len(avail)] = X_tr_raw
X_train_s = scaler.transform(X_tr_full)

# Load cross-building data
print("\nLoading cross-building data...")
cb_parts = []
for fname,lbl in CB_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        cb_parts.append(df)
        print(f"  {fname}: {len(df):,} rows | classes: {sorted(df['label'].unique())}")
df_cb  = pd.concat(cb_parts, ignore_index=True)
X_cb_s = to_X(df_cb, feat_cols, scaler)
y_cb   = df_cb['label'].values

cb_classes = sorted(np.unique(y_cb).tolist())
print(f"\n  Combined CB rows: {len(df_cb):,}")
print(f"  Classes present: {cb_classes}")
print(f"  Theoretical max macro-F1 (4-class): {len(cb_classes)/4:.2f}")
print(f"  (Classes 0 and 1 absent — dataset structural constraint)")

# Baseline
y_base = model.predict(X_cb_s)
f1_base_4  = f1_score(y_cb, y_base, average='macro', labels=[0,1,2,3], zero_division=0)
f1_base_2  = f1_score(y_cb, y_base, average='macro', labels=cb_classes, zero_division=0)
print(f"\n  BASELINE:")
print(f"    Macro-F1 (4-class incl absent): {f1_base_4:.4f}")
print(f"    Macro-F1 (present classes 2+3): {f1_base_2:.4f}")
print(f"    % of theoretical max (4-class): {f1_base_4/0.5*100:.1f}%")

# Fine-tuning
print(f"\n  FINE-TUNING (adapt on % of CB pool, eval on rest):")
print(f"  {'Pct':>5} {'N':>7} {'F1-4cls':>9} {'F1-2cls':>9} {'Chg-2cls':>10}")
print(f"  {'-'*5} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")

results = []
for pct in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-pct, random_state=RS)
    for ai,ei in sss.split(X_cb_s, y_cb):
        X_a,y_a = X_cb_s[ai],y_cb[ai]
        X_e,y_e = X_cb_s[ei],y_cb[ei]

    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                 class_weight='balanced', n_jobs=-1, random_state=RS)
    rf.fit(np.vstack([X_train_s,X_a]), np.concatenate([y_train,y_a]))
    y_pred = rf.predict(X_e)

    f1_4 = f1_score(y_e, y_pred, average='macro', labels=[0,1,2,3], zero_division=0)
    f1_2 = f1_score(y_e, y_pred, average='macro', labels=cb_classes, zero_division=0)
    chg  = (f1_2 - f1_base_2)*100
    print(f"  {pct*100:>4.0f}% {len(y_a):>7,} {f1_4:>9.4f} {f1_2:>9.4f} {chg:>+9.1f} pp")
    results.append(dict(pct=pct,n=int(len(y_a)),f1_4=round(f1_4,4),
                        f1_2=round(f1_2,4),chg=round(chg,1)))

best4 = max(results,key=lambda x:x['f1_4'])
best2 = max(results,key=lambda x:x['f1_2'])
print(f"\n  Best F1-4class: {best4['f1_4']:.4f} at {best4['pct']*100:.0f}% adaptation")
print(f"  Best F1-2class: {best2['f1_2']:.4f} at {best2['pct']*100:.0f}% adaptation")

out = dict(theoretical_max=0.50,
           baseline_f1_4class=round(f1_base_4,4),
           baseline_f1_2class=round(f1_base_2,4),
           results=results)
(BASE_DIR/'results').mkdir(exist_ok=True)
with open(BASE_DIR/'results'/'finetune_fixed_results.json','w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/finetune_fixed_results.json")
