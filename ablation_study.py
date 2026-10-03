"""
ABLATION STUDY — Fixed Version
================================
Fixes:
  FIX 1: Rolling windows computed per-file (inside load_file)
  FIX 2: No fabricated healthy data
  FIX 3: Stratified class balance enforced at each config

Run: python ablation_study.py
"""
import warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR = BASE_DIR / "data"
RS = 42
RAW_COLS = [
    'AHU: Supply Air Temperature','AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature', 'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status', 'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal','AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
TRAIN_FILES = {'MZVAV-1.csv':1,'MZVAV-2-1.csv':2,'SZCAV.csv':3}
CB_FILES    = {'MZVAV-2-2.csv':2,'SZVAV.csv':3}

def engineer(df):
    avail = [c for c in RAW_COLS if c in df.columns]
    out = df[avail].copy()
    for col in avail:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in avail:
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

print("="*62)
print("  ABLATION STUDY — Fixed Version")
print("="*62)

print("\nLoading training data...")
train_parts = []
for fname,lbl in TRAIN_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty:
        train_parts.append(df); print(f"  {fname}: {len(df):,} rows")
df_all   = pd.concat(train_parts, ignore_index=True)
all_cols = [c for c in df_all.columns if c!='label']

print("\nLoading cross-building data...")
cb_parts = []
for fname,lbl in CB_FILES.items():
    df = load_file(fname, lbl)
    if not df.empty: cb_parts.append(df)
df_cb = pd.concat(cb_parts, ignore_index=True) if cb_parts else pd.DataFrame()
print(f"  Cross-building: {len(df_cb):,} rows")

# Subsample
parts = []
for lbl in df_all['label'].unique():
    sub = df_all[df_all['label']==lbl]
    parts.append(sub.sample(min(len(sub),30000),random_state=RS))
df_sub = pd.concat(parts).reset_index(drop=True)

raw   = [c for c in all_cols if '_rm' not in c and '_rs' not in c and 'diff' not in c]
w10   = raw + [c for c in all_cols if '_rm10' in c or '_rs10' in c]
w1030 = raw + [c for c in all_cols if '_rm10' in c or '_rs10' in c or '_rm30' in c or '_rs30' in c]
allw  = [c for c in all_cols if 'diff' not in c]
full  = all_cols

configs = [
    ('Config 1 — Raw sensors only (11)',    raw),
    ('Config 2 — Raw + Window 10',          w10),
    ('Config 3 — Raw + Windows 10+30',      w1030),
    ('Config 4 — All windows, no derived',  allw),
    ('Config 5 — Full feature set (70)',    full),
]

print(f"\n  {'Config':<40} {'N':>5} {'In-samp F1':>11} {'Cross-bldg F1':>14}")
print(f"  {'-'*40} {'-'*5} {'-'*11} {'-'*14}")

results = []
for name, cols in configs:
    use = [c for c in cols if c in df_sub.columns]
    X = df_sub[use].fillna(0).values
    y = df_sub['label'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    rf  = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                  class_weight='balanced',n_jobs=-1,random_state=RS)
    rf.fit(Xtr,ytr)
    f1_in = f1_score(yte,rf.predict(Xte),average='macro',zero_division=0)
    f1_cb = None
    if not df_cb.empty:
        Xcb = np.zeros((len(df_cb),len(use)))
        for i,c in enumerate(use):
            if c in df_cb.columns:
                Xcb[:,i] = pd.to_numeric(df_cb[c],errors='coerce').fillna(0).values
        f1_cb = f1_score(df_cb['label'].values,rf.predict(sc.transform(Xcb)),
                         average='macro',zero_division=0)
    print(f"  {name:<40} {len(use):>5} {f1_in:>11.4f} "
          f"{str(round(f1_cb,4)) if f1_cb else 'N/A':>14}")
    results.append({'config':name,'n':len(use),'f1_in':round(f1_in,4),
                    'f1_cb':round(f1_cb,4) if f1_cb else None})

out = BASE_DIR/'results'
out.mkdir(exist_ok=True)
with open(out/'ablation_results.json','w') as f:
    json.dump(results,f,indent=2)
print(f"\nSaved to results/ablation_results.json")
print("="*62)
