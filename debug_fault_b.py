"""
DEBUG — What happens to Fault-B cross-building F1 specifically
when we add simulated Fault-B data?

Run: python debug_fault_b.py
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR  = BASE_DIR / "data"
SDAHU_DIR = DATA_DIR / "lbnl_sdahu" / "LBNL_FDD_Dataset_SDAHU"
RS = 42

SDAHU_COL_MAP = {
    'SA_TEMP':'AHU: Supply Air Temperature','OA_TEMP':'AHU: Outdoor Air Temperature',
    'MA_TEMP':'AHU: Mixed Air Temperature','RA_TEMP':'AHU: Return Air Temperature',
    'SF_SPD_DM':'AHU: Supply Air Fan Status','SF_CS':'AHU: Supply Air Fan Speed Control Signal',
    'OA_DMPR_DM':'AHU: Outdoor Air Damper Control Signal',
    'RA_DMPR_DM':'AHU: Return Air Damper Control Signal',
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Control Signal','SYS_CTL':'Occupancy Mode Indicator',
}
RAW_COLS = [
    'AHU: Supply Air Temperature','AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature','AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status','AHU: Supply Air Fan Speed Control Signal',
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
        out[col] = pd.to_numeric(out[col],errors='coerce').fillna(0)
    if 'AHU: Heating Coil Valve Control Signal' not in out.columns:
        out['AHU: Heating Coil Valve Control Signal'] = 0.0
    for col in RAW_COLS:
        if col not in out.columns: out[col] = 0.0
        out[col] = pd.to_numeric(out[col],errors='coerce').fillna(0)
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

def load_real(fname, label):
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

print("="*65)
print("  DEBUG — Fault-B Specific Cross-Building Analysis")
print("="*65)

# Load ASHRAE
print("\nLoading ASHRAE data...")
parts = []
for fname,lbl in TRAIN_FILES.items():
    df = load_real(fname,lbl)
    if not df.empty: parts.append(df)
df_ashrae = pd.concat(parts,ignore_index=True)
feat_cols = [c for c in df_ashrae.columns if c!='label']

# Load CB
cb_parts = []
for fname,lbl in CB_FILES.items():
    df = load_real(fname,lbl)
    if not df.empty: cb_parts.append(df)
df_cb = pd.concat(cb_parts,ignore_index=True)

def run_and_report(df_train, feat_cols, df_cb, label):
    use = [c for c in feat_cols if c in df_train.columns]
    # Balance
    min_n = min(df_train['label'].value_counts().min(), 15000)
    parts = [df_train[df_train['label']==l].sample(min_n,random_state=RS)
             for l in df_train['label'].unique()]
    df_sub = pd.concat(parts).reset_index(drop=True)
    X = df_sub[use].fillna(0).values
    y = df_sub['label'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    rf = RandomForestClassifier(n_estimators=100,max_features='sqrt',
                                 class_weight='balanced',n_jobs=-1,random_state=RS)
    rf.fit(Xtr_s,ytr)
    # CB evaluation — per class
    Xcb = np.zeros((len(df_cb),len(use)))
    for i,c in enumerate(use):
        if c in df_cb.columns:
            Xcb[:,i] = pd.to_numeric(df_cb[c],errors='coerce').fillna(0).values
    y_cb = df_cb['label'].values
    y_pred_cb = rf.predict(sc.transform(Xcb))
    f1_macro = f1_score(y_cb,y_pred_cb,average='macro',zero_division=0)
    # Per-class
    classes = sorted(np.unique(y_cb))
    per_class = {}
    for c in classes:
        mask = y_cb==c
        pass  # per-class handled by classification_report below
    report = classification_report(y_cb,y_pred_cb,output_dict=True,zero_division=0)
    print(f"\n  {label}")
    print(f"  Macro F1: {f1_macro:.4f}")
    for cls in [0,2,3]:
        key = str(cls)
        if key in report:
            r = report[key]
            cname = {0:'Healthy',2:'Fault-B',3:'Fault-C'}[cls]
            print(f"  {cname}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
    return f1_macro

# Baseline
f1_base = run_and_report(df_ashrae, feat_cols, df_cb, "ASHRAE only (baseline)")

# Add Fault-B simulated at different volumes
for n_rows in [5000, 10000, 25000, 50000]:
    sdahu_parts = []
    for fname in ['coi_leakage_010_annual.csv','coi_stuck_010_annual.csv']:
        path = SDAHU_DIR / fname
        if path.exists():
            df = pd.read_csv(path,nrows=n_rows).rename(columns=SDAHU_COL_MAP)
            fe = engineer(df); fe['label'] = 2
            sdahu_parts.append(fe)
    df_combined = pd.concat([df_ashrae]+sdahu_parts,ignore_index=True)
    use = [c for c in feat_cols if c in df_combined.columns]
    f1 = run_and_report(df_combined, use, df_cb,
                        f"+ Fault-B sim ({n_rows} rows/file, 2 files)")
    print(f"  Change vs baseline: {(f1-f1_base)*100:+.1f} pp")

print("\nConclusion: If all configs show CB F1 below baseline,")
print("the Fault-B simulated data is genuinely not helping.")
print("This is the finding to report — not a bug to fix.")
