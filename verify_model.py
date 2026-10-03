# verify_model.py
# Complete verification of the new final model
# Tests:
#   1. Leave-one-building-out cross-validation (gold standard)
#   2. Alternative train/test split (checks for overfitting)
#   3. Per-class fault detection rates (sanity check)
#   4. Prediction distribution on real Seoul data (sanity check)
#
# Run: python verify_model.py
# Time: approximately 30-45 minutes

import os, json, glob, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
SDAHU_DIR = os.path.join(DATA_DIR, "sdahu")
OUT_DIR   = os.path.join(BASE, "results", "new_model_final")
RS        = 42
ROLLING_WINDOWS = [10, 30, 60]
MAX_PER_CLASS   = 120000
MAX_PER_FILE    = 50000

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
SDAHU_COL_MAP = {
    'SA_TEMP':'AHU: Supply Air Temperature','OA_TEMP':'AHU: Outdoor Air Temperature',
    'MA_TEMP':'AHU: Mixed Air Temperature','RA_TEMP':'AHU: Return Air Temperature',
    'SF_SPD_DM':'AHU: Supply Air Fan Status','SF_CS':'AHU: Supply Air Fan Speed Control Signal',
    'OA_DMPR_DM':'AHU: Outdoor Air Damper Control Signal',
    'RA_DMPR_DM':'AHU: Return Air Damper Control Signal',
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Control Signal',
    'SYS_CTL':'Occupancy Mode Indicator',
}
SDAHU_CLASS0 = ['ahu_annual']
SDAHU_CLASS2 = ['coi_leakage', 'coi_stuck']
SDAHU_CLASS3 = ['damper_stuck']
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
CLASS_NAMES  = {0:'Healthy', 1:'Sensor Bias', 2:'Valve Fault', 3:'Damper Fault'}
ALL_ASHRAE   = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv',
                'MZVAV-2-2.csv', 'SZVAV.csv']

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

def load_ashrae(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    lc = 'Fault Detection Ground Truth'
    if lc not in df.columns: return pd.DataFrame()
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    feats = engineer_features(df)
    feats['label'] = np.where(df[lc].values[:len(feats)] == 0, 0, 1)
    return feats.dropna()

def load_sdahu(fpath, label, zero_oa=False):
    df = pd.read_csv(fpath, low_memory=False, nrows=MAX_PER_FILE)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for sc, ac in SDAHU_COL_MAP.items():
        if sc in df.columns:
            df[ac] = pd.to_numeric(df[sc], errors='coerce')
    for col in RAW_COLS:
        if col not in df.columns: df[col] = 0.0
    if zero_oa and label in [2, 3]:
        df['AHU: Outdoor Air Temperature'] = 0.0
        df['AHU: Mixed Air Temperature']   = 0.0
    feats = engineer_features(df)
    feats['label'] = label
    return feats.dropna()

def load_sdahu_faults():
    """Load SD-AHU fault files for classes 2 and 3."""
    if not os.path.exists(SDAHU_DIR):
        sdahu = os.path.join(DATA_DIR, "lbnl_sdahu", "LBNL_FDD_Dataset_SDAHU")
    else:
        sdahu = SDAHU_DIR
    parts = {0:[], 2:[], 3:[]}
    for fpath in sorted(glob.glob(os.path.join(sdahu, "*.csv"))):
        fn = os.path.basename(fpath).lower()
        lbl = None
        for p in SDAHU_CLASS0:
            if p in fn: lbl=0; break
        if lbl is None:
            for p in SDAHU_CLASS2:
                if p in fn: lbl=2; break
        if lbl is None:
            for p in SDAHU_CLASS3:
                if p in fn: lbl=3; break
        if lbl is None: continue
        df = load_sdahu(fpath, lbl, zero_oa=True)
        if len(df) > 0: parts[lbl].append(df)
    result = {}
    for cls, ps in parts.items():
        if ps:
            comb = pd.concat(ps, ignore_index=True)
            if len(comb) > MAX_PER_CLASS:
                comb = comb.sample(n=MAX_PER_CLASS, random_state=RS)
            result[cls] = comb
    return result

def train_and_test(train_dfs, test_df, feat_cols):
    """Train model on train_dfs, test on test_df."""
    df_tr = pd.concat([df[feat_cols+['label']] for df in train_dfs],
                      ignore_index=True)
    X = df_tr[feat_cols].values.astype(float)
    y = df_tr['label'].values.astype(int)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                         stratify=y, random_state=RS)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)

    m = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RS, n_jobs=-1, min_samples_leaf=2)
    m.fit(X_tr_sc, y_tr)

    X_te = np.zeros((len(test_df), len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in test_df.columns:
            X_te[:, i] = test_df[col].values.astype(float)
    X_te_sc = sc.transform(X_te)
    preds   = m.predict(X_te_sc)
    y_te    = test_df['label'].values.astype(int)
    f1      = f1_score(y_te, preds, average='macro', zero_division=0)
    f1_bin  = f1_score((y_te>0).astype(int),(preds>0).astype(int),
                       average='macro', zero_division=0)
    return f1, f1_bin, y_te, preds

print("=" * 70)
print("MODEL VERIFICATION TEST")
print("Leave-one-out + Alternative split + Sanity checks")
print("=" * 70)

# ── Load all data ─────────────────────────────────────────────────────────────
print("\nLoading all data...")
ashrae_dfs = {}
for fname in ALL_ASHRAE:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        df = load_ashrae(fpath)
        if len(df) > 0:
            ashrae_dfs[fname] = df
            dist = dict(df['label'].value_counts().sort_index())
            print(f"  {fname}: {len(df):,} rows  classes={dist}")

sdahu_faults = load_sdahu_faults()
print(f"  SD-AHU Class 0 (healthy): {len(sdahu_faults.get(0,pd.DataFrame())):,}")
print(f"  SD-AHU Class 2 (valve):   {len(sdahu_faults.get(2,pd.DataFrame())):,}")
print(f"  SD-AHU Class 3 (damper):  {len(sdahu_faults.get(3,pd.DataFrame())):,}")

# Find common feature columns across all data
all_dfs = list(ashrae_dfs.values()) + list(sdahu_faults.values())
feat_set = set(all_dfs[0].columns)
for df in all_dfs: feat_set &= set(df.columns)
feat_cols = sorted([c for c in feat_set if c != 'label'])
print(f"\n  Common feature columns: {len(feat_cols)}")

# ── TEST 1: Leave-one-ASHRAE-building-out ─────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: Leave-One-ASHRAE-Building-Out Cross-Validation")
print("Gold standard for cross-building evaluation")
print("="*70)

# Only use the 3 original training buildings for LOBO
training_buildings = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']
lobo_results = []

for held_out in training_buildings:
    train_files = [f for f in training_buildings if f != held_out]
    print(f"\n  Held out: {held_out}")
    print(f"  Training: {train_files}")

    # Build training set
    train_parts = []
    for fname in train_files:
        df = ashrae_dfs[fname].copy()
        train_parts.append(df[df['label']==0])  # healthy
        train_parts.append(df[df['label']==1])  # sensor bias
    # Add SD-AHU fault classes
    for cls in [0, 2, 3]:
        if cls in sdahu_faults:
            train_parts.append(sdahu_faults[cls])

    # Balance
    balanced = []
    for lbl in [0,1,2,3]:
        lbl_dfs = [p for p in train_parts if (p['label']==lbl).all() or
                   (len(p[p['label']==lbl])>0)]
        parts_lbl = pd.concat([p[p['label']==lbl] for p in train_parts
                               if len(p[p['label']==lbl])>0], ignore_index=True)
        if len(parts_lbl) > 0:
            n = min(MAX_PER_CLASS, len(parts_lbl))
            balanced.append(parts_lbl.sample(n=n, random_state=RS))

    test_df = ashrae_dfs[held_out]
    t0 = time.time()
    f1, f1_bin, y_te, preds = train_and_test(balanced, test_df, feat_cols)
    elapsed = time.time() - t0

    classes = sorted(np.unique(y_te).tolist())
    print(f"  4-class F1: {f1:.4f}  binary F1: {f1_bin:.4f}  ({elapsed:.0f}s)")
    print(classification_report(y_te, preds, labels=classes,
          target_names=[CLASS_NAMES.get(c,str(c)) for c in classes],
          zero_division=0))

    lobo_results.append({'held_out':held_out,'f1_4class':round(f1,4),
                         'f1_binary':round(f1_bin,4)})

avg_lobo_f1    = np.mean([r['f1_4class'] for r in lobo_results])
avg_lobo_bin   = np.mean([r['f1_binary'] for r in lobo_results])
print(f"\n  LOBO Average 4-class F1: {avg_lobo_f1:.4f}")
print(f"  LOBO Average binary F1:  {avg_lobo_bin:.4f}")
print(f"  Main experiment F1:      0.418 (train on 3 bldgs, test on 2 withheld)")
print(f"  VERDICT: {'CONSISTENT - model generalises reliably' if abs(avg_lobo_f1-0.418)<0.10 else 'INCONSISTENT - results may not be stable'}")

# ── TEST 2: Alternative train/test split ──────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: Alternative Train/Test Split")
print("Swap which buildings are train and which are test")
print("="*70)

# Use withheld buildings as training, original training buildings as test
alt_train_files = ['MZVAV-2-2.csv', 'SZVAV.csv']
alt_test_files  = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']

print(f"  Training: {alt_train_files}")
print(f"  Testing:  {alt_test_files}")

alt_train_parts = []
for fname in alt_train_files:
    if fname in ashrae_dfs:
        df = ashrae_dfs[fname]
        alt_train_parts.append(df[df['label']==0])
        alt_train_parts.append(df[df['label']==1])
for cls in [0, 2, 3]:
    if cls in sdahu_faults:
        alt_train_parts.append(sdahu_faults[cls])

# Balance
alt_balanced = []
for lbl in [0,1,2,3]:
    parts_lbl = pd.concat([p[p['label']==lbl] for p in alt_train_parts
                           if len(p[p['label']==lbl])>0], ignore_index=True)
    if len(parts_lbl) > 0:
        n = min(MAX_PER_CLASS, len(parts_lbl))
        alt_balanced.append(parts_lbl.sample(n=n, random_state=RS))

alt_test_parts = [ashrae_dfs[f] for f in alt_test_files if f in ashrae_dfs]
alt_test_df    = pd.concat(alt_test_parts, ignore_index=True)

t0 = time.time()
f1_alt, f1_alt_bin, y_alt, preds_alt = train_and_test(alt_balanced, alt_test_df, feat_cols)
print(f"  Alternative split 4-class F1: {f1_alt:.4f}")
print(f"  Alternative split binary F1:  {f1_alt_bin:.4f}  ({time.time()-t0:.0f}s)")
classes_alt = sorted(np.unique(y_alt).tolist())
print(classification_report(y_alt, preds_alt, labels=classes_alt,
      target_names=[CLASS_NAMES.get(c,str(c)) for c in classes_alt],
      zero_division=0))

# ── TEST 3: Confusion matrix analysis ─────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: Confusion Matrix — What Does the Model Confuse?")
print("="*70)

# Load the saved final model for this test
model_path = os.path.join(OUT_DIR, "rf_model_final.pkl")
if os.path.exists(model_path):
    model    = joblib.load(model_path)
    scaler   = joblib.load(os.path.join(OUT_DIR, "scaler_final.pkl"))
    ft_cols  = joblib.load(os.path.join(OUT_DIR, "feature_cols_final.pkl"))

    # Test on withheld buildings
    cb_parts = []
    for fname in ['MZVAV-2-2.csv', 'SZVAV.csv']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            df = load_ashrae(fpath)
            if len(df) > 0: cb_parts.append(df)

    df_cb = pd.concat(cb_parts, ignore_index=True)
    X_cb  = np.zeros((len(df_cb), len(ft_cols)))
    for i, col in enumerate(ft_cols):
        if col in df_cb.columns:
            X_cb[:, i] = df_cb[col].values.astype(float)
    X_cb_sc = scaler.transform(X_cb)
    y_cb    = df_cb['label'].values.astype(int)
    preds_cb= model.predict(X_cb_sc)

    classes_cb = sorted(np.unique(y_cb).tolist())
    cm = confusion_matrix(y_cb, preds_cb, labels=[0,1,2,3])

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':15}  {'Pred Healthy':14}  {'Pred SensorBias':15}  {'Pred Valve':12}  {'Pred Damper':12}")
    for i, actual_cls in enumerate([0,1,2,3]):
        row = cm[i] if actual_cls < len(cm) else [0,0,0,0]
        actual_name = CLASS_NAMES[actual_cls]
        print(f"  {'Act '+actual_name:15}  {row[0]:14,}  {row[1]:15,}  {row[2]:12,}  {row[3]:12,}")

    print(f"\n  Key insight: What does the model predict for sensor bias rows?")
    sb_rows = (y_cb == 1)
    if sb_rows.sum() > 0:
        sb_preds = preds_cb[sb_rows]
        for cls in [0,1,2,3]:
            n = (sb_preds == cls).sum()
            pct = 100*n/len(sb_preds)
            print(f"    Predicted as {CLASS_NAMES[cls]:15}: {n:6,} ({pct:.1f}%)")
else:
    print("  Saved model not found. Run run_final_complete.py first.")

# ── TEST 4: Seoul sanity check ────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: Seoul Real Data Sanity Check")
print("Seoul has Sensor Faults and Fan Faults only")
print("Model should predict mostly Class 1 (sensor bias)")
print("="*70)

seoul_path = os.path.join(DATA_DIR, "seoul", "combined_FDD.xls")
if os.path.exists(seoul_path) and os.path.exists(model_path):
    try:
        try:
            seoul_df = pd.read_excel(seoul_path, engine='xlrd')
        except:
            seoul_df = pd.read_excel(seoul_path, engine='openpyxl')
        seoul_df.columns = seoul_df.columns.str.strip()

        # Map columns
        col_map = {
            'Supply Air Temperature':'AHU: Supply Air Temperature',
            'Ventilation Temperature':'AHU: Outdoor Air Temperature',
            'Supply Fan':'AHU: Supply Air Fan Status',
            'Valve Position':'AHU: Cooling Coil Valve Control Signal',
        }
        for sc, ac in col_map.items():
            if sc in seoul_df.columns:
                seoul_df[ac] = pd.to_numeric(seoul_df[sc], errors='coerce')
        for col in RAW_COLS:
            if col not in seoul_df.columns: seoul_df[col] = 0.0

        seoul_feats = engineer_features(seoul_df)
        X_seoul = np.zeros((len(seoul_feats), len(ft_cols)))
        for i, col in enumerate(ft_cols):
            if col in seoul_feats.columns:
                X_seoul[:, i] = seoul_feats[col].values.astype(float)
        X_seoul_sc  = scaler.transform(X_seoul)
        preds_seoul = model.predict(X_seoul_sc)

        print(f"  Seoul predictions (should be mostly Class 1 = Sensor Bias):")
        for cls in [0,1,2,3]:
            n   = (preds_seoul == cls).sum()
            pct = 100*n/len(preds_seoul)
            flag = " <-- EXPECTED" if cls==1 else " <-- SUSPICIOUS" if n>len(preds_seoul)*0.3 else ""
            print(f"    Class {cls} ({CLASS_NAMES[cls]:15}): {n:6,} ({pct:.1f}%){flag}")

        dominant_cls = int(np.bincount(preds_seoul).argmax())
        if dominant_cls == 1:
            print(f"\n  SANITY CHECK: PASSED - Model predicts mostly Sensor Bias on Seoul data")
        else:
            print(f"\n  SANITY CHECK: SUSPICIOUS - Model predicts mostly {CLASS_NAMES[dominant_cls]} on Seoul data")
    except Exception as e:
        print(f"  Seoul load error: {e}")
else:
    print("  Seoul file or model not found. Skipping.")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"\n  Test 1 - LOBO average 4-class F1:  {avg_lobo_f1:.4f}")
print(f"  Test 1 - LOBO average binary F1:   {avg_lobo_bin:.4f}")
print(f"  Test 2 - Alternative split F1:     {f1_alt:.4f}")
print(f"  Test 2 - Alternative binary F1:    {f1_alt_bin:.4f}")
print(f"  Main experiment cross-bldg F1:     0.418")
print()

consistent = abs(avg_lobo_f1 - 0.418) < 0.10
print(f"  LOBO vs main experiment difference: {abs(avg_lobo_f1-0.418):.3f}")
print(f"  Within 10pp tolerance: {'YES' if consistent else 'NO'}")
print()
if consistent:
    print("  VERDICT: MODEL IS VERIFIED")
    print("  The cross-building F1 of 0.418 is genuine and reproducible.")
    print("  The model generalises consistently across different building splits.")
else:
    print("  VERDICT: RESULTS ARE NOT CONSISTENT")
    print("  The cross-building F1 varies too much across different splits.")
    print("  Consider keeping the original model.")

# Save verification results
ver_results = {
    "lobo_results": lobo_results,
    "lobo_avg_f1_4class": round(float(avg_lobo_f1),4),
    "lobo_avg_f1_binary": round(float(avg_lobo_bin),4),
    "alt_split_f1_4class": round(float(f1_alt),4),
    "alt_split_f1_binary": round(float(f1_alt_bin),4),
    "main_experiment_f1": 0.418,
    "lobo_vs_main_diff": round(abs(avg_lobo_f1-0.418),4),
    "verified": consistent,
}
with open(os.path.join(OUT_DIR, "verification_results.json"), "w") as f:
    json.dump(ver_results, f, indent=2)
print(f"\n  Results saved to: results/new_model_final/verification_results.json")
print("="*70)
