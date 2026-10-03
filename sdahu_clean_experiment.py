"""
SD-AHU Clean Cross-Building Experiment
Retrain on shared features only — no zero-filling distortion
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
import joblib
warnings.filterwarnings('ignore')

ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
SDAHU_DIR    = r'C:\Users\hrslp\Desktop\thesis\data\sdahu'
RANDOM_STATE = 42
SUBSAMPLE_N  = 30000
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]

# ── OFFICIAL SD-AHU COLUMN MAPPING ───────────────────────────────────────────
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

# Training files (your thesis training buildings)
TRAIN_FILES = {
    'MZVAV-1.csv' : None,   # label comes from Fault Detection Ground Truth
    'MZVAV-2-1.csv': None,
    'SZCAV.csv'   : None,
}

# SD-AHU fault files
SDAHU_FAULT_FILES = {
    0: ['AHU_annual.csv'],
    1: ['oa_bias_2_annual.csv', 'oa_bias_-2_annual.csv'],  # OA sensor bias
    2: ['coi_leakage_025_annual.csv'],                       # Valve leakage
}

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer(df, shared_cols):
    """Engineer features using only shared columns."""
    raw = [c for c in shared_cols if c in df.columns]
    frames = [df[raw].copy()]
    for w in WINDOW_SIZES:
        rm = df[raw].rolling(w, min_periods=1).mean()
        rs = df[raw].rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    feat = pd.concat(frames, axis=1)

    # Derived — only from available columns
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

# ── LOAD ASHRAE TRAINING DATA ─────────────────────────────────────────────────

def load_ashrae_training():
    """Load your existing thesis training files."""
    frames = []
    for fname in TRAIN_FILES:
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found at {fpath}")
            continue
        df = pd.read_csv(fpath, low_memory=False)
        # Use ground truth label column (thesis label fix)
        if 'Fault Detection Ground Truth' in df.columns:
            # Class 0 = healthy, rest = fault class based on filename
            if 'MZVAV-1' in fname:   fault_class = 1
            elif 'MZVAV-2-1' in fname: fault_class = 2
            elif 'SZCAV' in fname:   fault_class = 3
            else:                     fault_class = 1
            df['label'] = np.where(
                df['Fault Detection Ground Truth'] == 0, 0, fault_class)
        else:
            df['label'] = 0
        df = df.select_dtypes(include=[np.number, object])
        frames.append(df)
        print(f"  {fname}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True) if frames else None

# ── LOAD SD-AHU DATA ──────────────────────────────────────────────────────────

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
            df = df.select_dtypes(include=[np.number])
            df['label'] = label
            frames.append(df)
            print(f"  {fname}: {len(df):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU CLEAN CROSS-BUILDING EXPERIMENT")
    print("Shared features only — no zero-fill distortion")
    print("=" * 65)

    # Load SD-AHU first to find its columns
    print("\nStep 1 — Loading SD-AHU data...")
    sdahu = load_sdahu()
    sdahu_cols = set(sdahu.columns) - {'label'}
    print(f"  SD-AHU columns available: {len(sdahu_cols)}")

    # Load ASHRAE training data
    print("\nStep 2 — Loading ASHRAE training data...")
    ashrae = load_ashrae_training()
    if ashrae is None:
        print("  ERROR: Could not load ASHRAE training files.")
        print("  Make sure MZVAV-1.csv, MZVAV-2-1.csv, SZCAV.csv are in data/")
        return
    ashrae_cols = set(ashrae.columns) - {'label', 'Fault Detection Ground Truth'}

    # Find shared columns
    shared = sorted(sdahu_cols & ashrae_cols)
    print(f"\nStep 3 — Finding shared sensor columns...")
    print(f"  ASHRAE columns: {len(ashrae_cols)}")
    print(f"  SD-AHU columns: {len(sdahu_cols)}")
    print(f"  Shared columns: {len(shared)}")
    print(f"  Shared: {shared}")

    # Engineer features on shared columns only
    print("\nStep 4 — Engineering features on shared columns...")
    # ASHRAE
    ashrae_feat = engineer(ashrae[shared + ['label']], shared)
    ashrae_feat['label'] = ashrae['label'].values
    # SD-AHU
    sdahu_feat  = engineer(sdahu[shared + ['label']], shared)
    sdahu_feat['label']  = sdahu['label'].values

    feat_cols = [c for c in ashrae_feat.columns if c != 'label']
    print(f"  Total features engineered: {len(feat_cols)}")

    # Prepare ASHRAE training set
    y_train = ashrae_feat['label'].values
    X_train = ashrae_feat[feat_cols]

    # Subsample for balance
    print("\nStep 5 — Subsampling and training model...")
    frames_sub = []
    for cls in np.unique(y_train):
        idx = np.where(y_train == cls)[0]
        n   = min(len(idx), SUBSAMPLE_N)
        chosen = np.random.default_rng(RANDOM_STATE).choice(idx, n, replace=False)
        frames_sub.append((X_train.iloc[chosen], y_train[chosen]))
    X_tr = pd.concat([f[0] for f in frames_sub])
    y_tr = np.concatenate([f[1] for f in frames_sub])
    print(f"  Training set: {len(y_tr):,} rows")
    print(f"  Class distribution: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    # Train model
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_tr_scaled, y_tr)
    y_pred_train = clf.predict(X_tr_scaled)
    f1_train = f1_score(y_tr, y_pred_train, average='macro', zero_division=0)
    print(f"  In-sample macro-F1: {f1_train:.4f}")

    # Prepare SD-AHU test set
    print("\nStep 6 — Cross-building evaluation on SD-AHU...")
    y_test = sdahu_feat['label'].values
    X_test = sdahu_feat[feat_cols]

    # Scale using training scaler (frozen)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate — 3 classes only (Healthy, Fault-A, Fault-B)
    mask3 = np.isin(y_test, [0, 1, 2])
    X3, y3 = X_test_scaled[mask3], y_test[mask3]
    np.random.seed(RANDOM_STATE)
    idx3 = np.random.choice(len(y3), min(len(y3), 15000), replace=False)
    yp3  = clf.predict(X3[idx3])
    f1_3 = f1_score(y3[idx3], yp3, average='macro', zero_division=0)

    print(f"\n  3-class macro-F1: {f1_3:.4f}")
    print(classification_report(y3[idx3], yp3,
          target_names=['Healthy','Fault-A (OA bias)','Fault-B (valve)'],
          zero_division=0))

    # PSI analysis on shared features
    print("\nStep 7 — PSI analysis (shared features only)...")
    psi_scores = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for i, col in enumerate(feat_cols):
        ref = X_tr_scaled[:, i]
        tgt = X_test_scaled[:, i]
        psi_scores[col] = compute_psi(ref, tgt)

    psi_s = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 PSI features (no zero-fill distortion):")
    for fn, psi in psi_s.head(10).items():
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn[:50]:<50} PSI={psi:.3f}  {tag}")

    # Raw sensor PSI
    print("\n  Raw sensor PSI values:")
    raw_psi = {k: v for k, v in psi_scores.items()
               if '_rm' not in k and '_rs' not in k
               and 'diff' not in k and 'ratio' not in k}
    for fn, psi in sorted(raw_psi.items(), key=lambda x: -x[1]):
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn[:50]:<50} PSI={psi:.3f}  {tag}")

    print("\n" + "=" * 65)
    print("FINAL RESULTS — CLEAN EXPERIMENT")
    print("=" * 65)
    print(f"  Shared features used:                    {len(feat_cols)}")
    print(f"  New model in-sample F1:                  {f1_train:.4f}")
    print(f"  Thesis original in-sample F1:            0.9923")
    print(f"  Thesis CB F1 (SZVAV + MZVAV-2-2):        0.331")
    print(f"  SD-AHU CB F1 (clean, shared features):   {f1_3:.4f}")
    print()

    gap = f1_3 - 0.331
    if   gap >  0.05: verdict = "BETTER  — gap smaller on SD-AHU building"
    elif gap < -0.05: verdict = "WORSE   — gap larger on SD-AHU building"
    else:             verdict = "SIMILAR — ~0.33 gap reproduced on third building"

    print(f"  Verdict: {verdict}")
    print()
    print("  What this means for the thesis:")
    if gap < -0.05:
        print("  The cross-building generalisation failure is even more severe")
        print("  on this building type. The 0.331 thesis result was not the")
        print("  worst case — generalisation failure is a systematic pattern.")
        print("  This STRENGTHENS Contribution C1.")
    elif gap > 0.05:
        print("  Some buildings show smaller generalisation gaps than others.")
        print("  NEW FINDING: the gap magnitude varies by building type.")
        print("  PSI analysis above explains which features drive the difference.")
    else:
        print("  The ~0.33 cross-building gap is REPRODUCIBLE on a third building.")
        print("  This directly strengthens Contribution C1 — the gap is not")
        print("  specific to one building pair but a systematic pattern.")

    results = {
        'experiment'         : 'sdahu_clean_shared_features',
        'shared_features'    : len(feat_cols),
        'f1_insample'        : round(float(f1_train), 4),
        'f1_cb_3class'       : round(float(f1_3),     4),
        'thesis_cb_f1'       : 0.331,
        'gap_vs_thesis'      : round(float(gap),       4),
        'verdict'            : verdict,
        'top5_psi_clean'     : psi_s.head(5).round(3).to_dict(),
        'raw_sensor_psi'     : {k: round(v,3) for k,v in raw_psi.items()},
    }
    os.makedirs('results', exist_ok=True)
    with open('results/sdahu_clean_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to results/sdahu_clean_experiment.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
