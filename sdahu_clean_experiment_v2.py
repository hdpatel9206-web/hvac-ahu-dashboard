"""
SD-AHU Clean Cross-Building Experiment v2
Fixed: #VALUE! cells cleaned before feature engineering
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import joblib
warnings.filterwarnings('ignore')

# ── PATHS — update these if needed ───────────────────────────────────────────
ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
SDAHU_DIR    = r'C:\Users\hrslp\Desktop\thesis\data\sdahu'
RANDOM_STATE = 42
SUBSAMPLE_N  = 30000
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]

# ── COLUMN MAPPING ────────────────────────────────────────────────────────────
COL_MAP = {
    'SF_CS'      : 'AHU: Supply Air Fan Speed Control Signal',
    'SF_SPD'     : 'AHU: Supply Air Fan Speed Position',
    'SF_WAT'     : 'AHU: Supply Air Fan Power',
    'RF_CS'      : 'AHU: Return Air Fan Speed Control Signal',
    'RF_SPD'     : 'AHU: Return Air Fan Speed Position',
    'RF_WAT'     : 'AHU: Return Air Fan Power',
    'SA_TEMP'    : 'AHU: Supply Air Temperature',
    'SA_TEMPSPT' : 'AHU: Supply Air Temperature Setpoint',
    'SA_CFM'     : 'AHU: Supply Air Flow Rate',
    'SA_SP'      : 'AHU: Supply Air Duct Static Pressure',
    'SA_SPSPT'   : 'AHU: Supply Air Duct Static Pressure Setpoint',
    'RA_TEMP'    : 'AHU: Return Air Temperature',
    'RA_CFM'     : 'AHU: Return Air Flow Rate',
    'RA_DMPR'    : 'AHU: Return Air Damper Position',
    'OA_TEMP'    : 'AHU: Outdoor Air Temperature',
    'OA_CFM'     : 'AHU: Outdoor Air Flow Rate',
    'OA_DMPR'    : 'AHU: Outdoor Air Damper Position',
    'MA_TEMP'    : 'AHU: Mixed Air Temperature',
    'CHWC_VLV'   : 'AHU: Cooling Coil Valve Position',
    'SYS_CTL'    : 'AHU: System Control Mode',
    'ZONE_TEMP_1': 'AHU: Zone Temperature 1',
    'ZONE_TEMP_2': 'AHU: Zone Temperature 2',
    'ZONE_TEMP_3': 'AHU: Zone Temperature 3',
    'ZONE_TEMP_4': 'AHU: Zone Temperature 4',
    'ZONE_TEMP_5': 'AHU: Zone Temperature 5',
}

SDAHU_FAULT_FILES = {
    0: ['AHU_annual.csv'],
    1: ['oa_bias_2_annual.csv', 'oa_bias_-2_annual.csv'],
    2: ['coi_leakage_025_annual.csv'],
}

TRAIN_FILES = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_df(df):
    """Replace #VALUE! and other Excel errors with NaN, then coerce to numeric."""
    # Replace common Excel error strings
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    # Coerce all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df, shared_cols):
    raw = [c for c in shared_cols if c in df.columns]
    # Clean before rolling
    df[raw] = df[raw].apply(pd.to_numeric, errors='coerce').fillna(0)
    frames = [df[raw].copy()]
    for w in WINDOW_SIZES:
        rm = df[raw].rolling(w, min_periods=1).mean()
        rs = df[raw].rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    feat = pd.concat(frames, axis=1)

    sa = 'AHU: Supply Air Temperature'
    ra = 'AHU: Return Air Temperature'
    oa = 'AHU: Outdoor Air Temperature'
    ma = 'AHU: Mixed Air Temperature'

    if sa in df.columns and ra in df.columns:
        feat['Temp_supply_return_diff'] = df[sa] - df[ra]
    if oa in df.columns and sa in df.columns:
        feat['Temp_oa_supply_diff']     = df[oa] - df[sa]
    if ma in df.columns and ra in df.columns:
        feat['Temp_mixed_return_diff']  = df[ma] - df[ra]

    return feat.fillna(0)

def compute_psi(ref, tgt, n_bins=10, eps=1e-6):
    lo = min(ref.min(), tgt.min())
    hi = max(ref.max(), tgt.max())
    if lo == hi: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    rp = np.histogram(ref, bins=bins)[0].astype(float) + eps
    tp = np.histogram(tgt, bins=bins)[0].astype(float) + eps
    rp /= rp.sum(); tp /= tp.sum()
    return float(np.sum((tp - rp) * np.log(tp / rp)))

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_ashrae():
    frames = []
    fault_map = {'MZVAV-1.csv': 1, 'MZVAV-2-1.csv': 2, 'SZCAV.csv': 3}
    for fname in TRAIN_FILES:
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df = clean_df(df)                          # ← fixes #VALUE! here
        if 'Fault Detection Ground Truth' in df.columns:
            df['label'] = np.where(
                df['Fault Detection Ground Truth'] == 0,
                0, fault_map.get(fname, 1))
        else:
            df['label'] = 0
        frames.append(df)
        n_fault = (df['label'] > 0).sum()
        print(f"  {fname}: {len(df):,} rows  "
              f"(healthy={( df['label']==0).sum():,}  fault={n_fault:,})")
    return pd.concat(frames, ignore_index=True) if frames else None

def load_sdahu():
    frames = []
    for label, files in SDAHU_FAULT_FILES.items():
        for fname in files:
            fpath = os.path.join(SDAHU_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found"); continue
            df = pd.read_csv(fpath, low_memory=False)
            df = df.rename(columns=COL_MAP)
            df = df.drop(columns=[c for c in df.columns
                         if 'datetime' in c.lower()], errors='ignore')
            df = clean_df(df)                      # ← clean here too
            df['label'] = label
            frames.append(df)
            print(f"  {fname}: {len(df):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU CLEAN CROSS-BUILDING EXPERIMENT v2")
    print("Shared features · #VALUE! cells cleaned")
    print("=" * 65)

    print("\nStep 1 — Loading SD-AHU data...")
    sdahu = load_sdahu()
    sdahu_cols = set(sdahu.columns) - {'label'}

    print("\nStep 2 — Loading ASHRAE training data...")
    ashrae = load_ashrae()
    if ashrae is None:
        print("ERROR: ASHRAE files not found. Check ASHRAE_DIR path."); return
    ashrae_cols = set(ashrae.columns) - {'label','Fault Detection Ground Truth'}

    print("\nStep 3 — Shared columns...")
    shared = sorted(sdahu_cols & ashrae_cols)
    print(f"  ASHRAE: {len(ashrae_cols)}  SD-AHU: {len(sdahu_cols)}  Shared: {len(shared)}")
    for c in shared:
        print(f"    ✓ {c}")

    print("\nStep 4 — Engineering features...")
    ashrae_feat = engineer(ashrae[shared + ['label']].copy(), shared)
    ashrae_feat['label'] = ashrae['label'].values
    sdahu_feat  = engineer(sdahu[shared  + ['label']].copy(), shared)
    sdahu_feat['label']  = sdahu['label'].values

    feat_cols = [c for c in ashrae_feat.columns if c != 'label']
    print(f"  Total features: {len(feat_cols)}")

    print("\nStep 5 — Training model on shared features...")
    y_tr_all = ashrae_feat['label'].values
    X_tr_all = ashrae_feat[feat_cols]

    # Subsample per class for balance
    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr_all):
        idx = np.where(y_tr_all == cls)[0]
        n   = min(len(idx), SUBSAMPLE_N)
        chosen = rng.choice(idx, n, replace=False)
        sub_X.append(X_tr_all.iloc[chosen])
        sub_y.append(y_tr_all[chosen])
    X_tr = pd.concat(sub_X)
    y_tr = np.concatenate(sub_y)
    print(f"  Training: {len(y_tr):,} rows  classes: {np.unique(y_tr)}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_tr_sc, y_tr)
    f1_train = f1_score(clf.predict(X_tr_sc), y_tr,
                        average='macro', zero_division=0)
    print(f"  In-sample macro-F1: {f1_train:.4f}")

    print("\nStep 6 — Cross-building evaluation on SD-AHU...")
    y_te = sdahu_feat['label'].values
    X_te = sdahu_feat[feat_cols]
    X_te_sc = scaler.transform(X_te)

    # 3-class (Healthy, Fault-A, Fault-B)
    mask3 = np.isin(y_te, [0, 1, 2])
    X3, y3 = X_te_sc[mask3], y_te[mask3]
    np.random.seed(RANDOM_STATE)
    idx3 = np.random.choice(len(y3), min(len(y3), 15000), replace=False)
    yp3  = clf.predict(X3[idx3])
    f1_3 = f1_score(y3[idx3], yp3, average='macro', zero_division=0)

    print(f"\n  3-class macro-F1: {f1_3:.4f}")
    print(classification_report(y3[idx3], yp3,
          target_names=['Healthy','Fault-A (OA bias)','Fault-B (valve)'],
          zero_division=0))

    print("Step 7 — PSI on shared features (clean)...")
    psi_scores = {}
    for i, col in enumerate(feat_cols):
        psi_scores[col] = compute_psi(X_tr_sc[:, i], X_te_sc[:, i])

    psi_s = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 PSI features:")
    for fn, psi in psi_s.head(10).items():
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn[:50]:<50} PSI={psi:.3f}  {tag}")

    raw_psi = {k: v for k, v in psi_scores.items()
               if '_rm' not in k and '_rs' not in k
               and 'diff' not in k and 'ratio' not in k}
    print("\n  Raw sensor PSI:")
    for fn, psi in sorted(raw_psi.items(), key=lambda x: -x[1]):
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn:<50} PSI={psi:.3f}  {tag}")

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  Shared features:                         {len(feat_cols)}")
    print(f"  In-sample F1 (retrained on shared):      {f1_train:.4f}")
    print(f"  Thesis in-sample F1 (all 70 features):   0.9923")
    print(f"  Thesis CB F1 (SZVAV + MZVAV-2-2):        0.331")
    print(f"  SD-AHU CB F1 (clean shared features):    {f1_3:.4f}")
    print()

    gap = f1_3 - 0.331
    if   gap >  0.05: v = "BETTER  — smaller gap on SD-AHU"
    elif gap < -0.05: v = "WORSE   — larger gap on SD-AHU"
    else:             v = "SIMILAR — ~0.33 gap reproduced on third building"
    print(f"  Verdict: {v}")

    results = {
        'experiment'      : 'sdahu_clean_v2',
        'shared_features' : len(feat_cols),
        'f1_insample'     : round(float(f1_train), 4),
        'f1_cb_3class'    : round(float(f1_3),     4),
        'thesis_cb_f1'    : 0.331,
        'gap'             : round(float(gap),       4),
        'verdict'         : v,
        'top5_psi'        : psi_s.head(5).round(3).to_dict(),
        'raw_sensor_psi'  : {k: round(v,3) for k,v in raw_psi.items()},
    }
    os.makedirs('results', exist_ok=True)
    with open('results/sdahu_clean_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: results/sdahu_clean_v2.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
