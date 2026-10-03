"""
DIAGNOSTIC — Which LBNL fault class hurts cross-building performance?
Tests adding each fault class incrementally to find the culprit.

Run: python diagnose_extended.py
"""
import warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR  = BASE_DIR / "data"
SDAHU_DIR = DATA_DIR / "lbnl_sdahu" / "LBNL_FDD_Dataset_SDAHU"
RS = 42

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

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
TRAIN_FILES = {'MZVAV-1.csv':1,'MZVAV-2-1.csv':2,'SZCAV.csv':3}
CB_FILES    = {'MZVAV-2-2.csv':2,'SZVAV.csv':3}

# One representative file per fault class
SDAHU_TEST_FILES = {
    'Fault-A only (oa_bias_2)':    [('oa_bias_2_annual.csv', 1)],
    'Fault-B only (coi_leakage)':  [('coi_leakage_010_annual.csv', 2)],
    'Fault-C only (damper_stuck)': [('damper_stuck_010_annual.csv', 3)],
    'Fault-A (8 files)': [(f,1) for f in ['oa_bias_2_annual.csv','oa_bias_4_annual.csv',
                           'oa_bias_-2_annual.csv','oa_bias_-4_annual.csv',
                           'coi_bias_2_annual.csv','coi_bias_4_annual.csv',
                           'coi_bias_-2_annual.csv','coi_bias_-4_annual.csv']],
    'Fault-B (8 files)': [(f,2) for f in ['coi_leakage_010_annual.csv','coi_leakage_025_annual.csv',
                           'coi_leakage_040_annual.csv','coi_leakage_050_annual.csv',
                           'coi_stuck_010_annual.csv','coi_stuck_025_annual.csv',
                           'coi_stuck_050_annual.csv','coi_stuck_075_annual.csv']],
    'Fault-C (4 files)': [(f,3) for f in ['damper_stuck_010_annual.csv','damper_stuck_025_annual.csv',
                           'damper_stuck_075_annual.csv','damper_stuck_100_annual_short.csv']],
}

def engineer(df):
    avail = [c for c in RAW_COLS if c in df.columns]
    out = df[avail].copy()
    for col in avail:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    if 'AHU: Heating Coil Valve Control Signal' not in out.columns:
        out['AHU: Heating Coil Valve Control Signal'] = 0.0
    for col in RAW_COLS:
        if col not in out.columns: out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in RAW_COLS:
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

def load_ashrae(fname, label):
    path = DATA_DIR / fname
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    lc = 'Fault Detection Ground Truth' if 'Fault Detection Ground Truth' in df.columns else \
         'Active Fault' if 'Active Fault' in df.columns else None
    if lc:
        h = df[df[lc]==0].copy(); f = df[df[lc]!=0].copy()
    else:
        h = pd.DataFrame(); f = df.copy()
    parts = []
    if not f.empty:
        fe = engineer(f); fe['label'] = label; parts.append(fe)
    if not h.empty:
        fe = engineer(h); fe['label'] = 0; parts.append(fe)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

print("="*62)
print("  DIAGNOSTIC — Which LBNL fault class hurts cross-building?")
print("="*62)

# Load ASHRAE baseline
print("\nLoading ASHRAE baseline...")
ashrae_parts = []
for fname,lbl in TRAIN_FILES.items():
    df = load_ashrae(fname, lbl)
    if not df.empty: ashrae_parts.append(df)
df_ashrae = pd.concat(ashrae_parts, ignore_index=True)
feat_cols = [c for c in df_ashrae.columns if c != 'label']

# Load CB test
cb_parts = []
for fname,lbl in CB_FILES.items():
    df = load_ashrae(fname, lbl)
    if not df.empty: cb_parts.append(df)
df_cb = pd.concat(cb_parts, ignore_index=True)

def run_experiment(df_train, feat_cols, df_cb, label=''):
    # Subsample balanced
    parts = []
    for lbl in df_train['label'].unique():
        s = df_train[df_train['label']==lbl]
        parts.append(s.sample(min(len(s),15000),random_state=RS))
    df_sub = pd.concat(parts).reset_index(drop=True)
    # Balance
    min_n = df_sub['label'].value_counts().min()
    balanced = pd.concat([df_sub[df_sub['label']==l].sample(min_n,random_state=RS)
                          for l in df_sub['label'].unique()]).reset_index(drop=True)
    use = [c for c in feat_cols if c in balanced.columns]
    X = balanced[use].fillna(0).values
    y = balanced['label'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    rf = RandomForestClassifier(n_estimators=100,max_features='sqrt',
                                 class_weight='balanced',n_jobs=-1,random_state=RS)
    rf.fit(Xtr_s,ytr)
    f1_in = f1_score(yte,rf.predict(Xte_s),average='macro',zero_division=0)
    Xcb = np.zeros((len(df_cb),len(use)))
    for i,c in enumerate(use):
        if c in df_cb.columns:
            Xcb[:,i] = pd.to_numeric(df_cb[c],errors='coerce').fillna(0).values
    f1_cb = f1_score(df_cb['label'].values,rf.predict(sc.transform(Xcb)),
                     average='macro',zero_division=0)
    return f1_in, f1_cb

print(f"\n  {'Experiment':<40} {'In-samp F1':>11} {'Cross-bldg F1':>14} {'Change':>8}")
print(f"  {'-'*40} {'-'*11} {'-'*14} {'-'*8}")

# Baseline: ASHRAE only
f1_base_in, f1_base_cb = run_experiment(df_ashrae, feat_cols, df_cb)
print(f"  {'ASHRAE only (baseline)':<40} {f1_base_in:>11.4f} {f1_base_cb:>14.4f} {'—':>8}")

# Add each SDAHU fault group
for config_name, file_list in SDAHU_TEST_FILES.items():
    sdahu_rows = []
    for fname, lbl in file_list:
        path = SDAHU_DIR / fname
        if path.exists():
            df = pd.read_csv(path, nrows=50000).rename(columns=SDAHU_COL_MAP)
            fe = engineer(df); fe['label'] = lbl
            sdahu_rows.append(fe)
    if not sdahu_rows: continue
    df_sdahu = pd.concat(sdahu_rows, ignore_index=True)
    df_combined = pd.concat([df_ashrae, df_sdahu], ignore_index=True)
    use_cols = [c for c in feat_cols if c in df_combined.columns]
    f1_in, f1_cb = run_experiment(df_combined, use_cols, df_cb)
    chg = (f1_cb - f1_base_cb)*100
    marker = ' ★' if f1_cb > f1_base_cb else ' ✗' if chg < -2 else ''
    print(f"  {'ASHRAE + ' + config_name:<40} {f1_in:>11.4f} {f1_cb:>14.4f} {chg:>+7.1f} pp{marker}")

print(f"\n  ★ = improves cross-building | ✗ = hurts cross-building")
