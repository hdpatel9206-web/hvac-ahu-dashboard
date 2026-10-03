"""
FCU Cross-Equipment Experiment
AHU-trained model tested on Fan Coil Unit (hotel/hospital/office)
FCU shares 8 physical sensor columns with AHU — best overlap so far
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

ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
FCU_DIR      = r'C:\Users\hrslp\Desktop\thesis\data\fcu'
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

# 8 shared physical sensor columns — confirmed by column analysis
COL_MAP = {
    'FCU_CVLV' : 'AHU: Cooling Coil Valve Control Signal',
    'FCU_HVLV' : 'AHU: Heating Coil Valve Control Signal',
    'FCU_MAT'  : 'AHU: Mixed Air Temperature',
    'FCU_DMPR' : 'AHU: Outdoor Air Damper Control Signal',
    'FCU_OAT'  : 'AHU: Outdoor Air Temperature',
    'FCU_RAT'  : 'AHU: Return Air Temperature',
    'FCU_SPD'  : 'AHU: Supply Air Fan Speed Control Signal',
    'FCU_DAT'  : 'AHU: Supply Air Temperature',
}

SHARED = list(COL_MAP.values())

# FCU fault files grouped by class
# Using one representative file per severity to keep runtime manageable
FCU_FILES = {
    0: ['FCU_FaultFree.csv'],
    1: ['FCU_Control_CoolingReverse.csv',
        'FCU_Control_HeatingReverse.csv',
        'FCU_Control_Unstable.csv'],
    2: ['FCU_FilterRestriction_10%.csv',
        'FCU_FilterRestriction_20%.csv'],
    3: ['Cooling_Waterside_Minor.csv',
        'Cooling_Waterside_Moderate.csv',
        'Cooling_Airside_Minor.csv'],
    4: ['Heating_Airside_Minor.csv',
        'Heating_Airside_Moderate.csv'],
    5: ['FCU_OABlockage.csv',
        'OADMPRLeak_20.csv',
        'OADMPRStuck_0.csv'],
    6: ['FCU_SensorBias_RMTemp_+2C.csv',
        'FCU_SensorBias_RMTemp_-2C.csv'],
    7: ['CoolingLeak_20.csv',
        'CoolingLeak_50.csv',
        'Stuck_Cooling_0.csv'],
    8: ['HeatingLeak_20.csv',
        'Stuck_Heating_0.csv'],
    9: ['FCU_FanOutletBlockage.csv'],
}

TRAIN_FILES = {
    'MZVAV-1.csv': 1, 'MZVAV-2-1.csv': 2, 'SZCAV.csv': 3
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def clean(df):
    df.columns = df.columns.str.strip()
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df):
    raw = [c for c in SHARED if c in df.columns]
    data = df[raw].copy().fillna(0)
    frames = [data]
    for w in WINDOW_SIZES:
        rm = data.rolling(w, min_periods=1).mean()
        rs = data.rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    feat = pd.concat(frames, axis=1)

    # Derived temperature features
    sa = 'AHU: Supply Air Temperature'
    ra = 'AHU: Return Air Temperature'
    oa = 'AHU: Outdoor Air Temperature'
    ma = 'AHU: Mixed Air Temperature'
    if sa in df.columns and ra in df.columns:
        feat['diff_sa_ra'] = df[sa].values - df[ra].values
    if oa in df.columns and sa in df.columns:
        feat['diff_oa_sa'] = df[oa].values - df[sa].values
    if ma in df.columns and ra in df.columns:
        feat['diff_ma_ra'] = df[ma].values - df[ra].values

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

def load_ashrae():
    frames = []
    for fname, fcls in TRAIN_FILES.items():
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = clean(pd.read_csv(fpath, low_memory=False))
        label = np.where(
            df.get('Fault Detection Ground Truth',
                   pd.Series(0, index=df.index)) == 0, 0, fcls)
        feat = engineer(df)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows")
    return pd.concat(frames, ignore_index=True)

def load_fcu():
    frames = []
    for label, files in FCU_FILES.items():
        for fname in files:
            fpath = os.path.join(FCU_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found — skipping")
                continue
            df = clean(pd.read_csv(fpath, low_memory=False))
            df = df.rename(columns=COL_MAP)
            feat = engineer(df)
            feat['label'] = label
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FCU CROSS-EQUIPMENT EXPERIMENT")
    print("AHU-trained model vs Fan Coil Unit")
    print("8 shared physical sensor columns")
    print("=" * 65)

    print(f"\nShared columns ({len(SHARED)}):")
    for c in SHARED: print(f"  ✓ {c}")

    print("\nLoading ASHRAE training data...")
    ashrae = load_ashrae()
    print("\nLoading FCU data...")
    fcu = load_fcu()

    feat_cols = [c for c in ashrae.columns if c != 'label']
    print(f"\nTotal features: {len(feat_cols)}")
    print(f"ASHRAE rows:    {len(ashrae):,}")
    print(f"FCU rows:       {len(fcu):,}")
    print(f"FCU classes:    {sorted(fcu['label'].unique())}")

    # Train base model
    print("\nTraining base model on ASHRAE...")
    y_tr = ashrae['label'].values
    X_tr = ashrae[feat_cols].values

    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr):
        idx = np.where(y_tr == cls)[0]
        n   = min(len(idx), 30000)
        ch  = rng.choice(idx, n, replace=False)
        sub_X.append(X_tr[ch]); sub_y.append(y_tr[ch])
    X_base = np.vstack(sub_X); y_base = np.concatenate(sub_y)

    scaler    = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_base_sc, y_base)
    f1_is = f1_score(clf.predict(X_base_sc), y_base,
                     average='macro', zero_division=0)
    print(f"  In-sample F1: {f1_is:.4f}")

    # FCU evaluation
    y_fc = fcu['label'].values
    X_fc = fcu[feat_cols].values
    X_fc_sc = scaler.transform(X_fc)

    # PSI on raw shared sensors
    print("\nPSI Analysis (raw sensors):")
    raw_cols = [c for c in feat_cols
                if '_rm' not in c and '_rs' not in c and 'diff' not in c]
    for i, col in enumerate(feat_cols):
        if col in raw_cols:
            psi = compute_psi(X_base_sc[:, i], X_fc_sc[:, i])
            tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
            print(f"  {col[:48]:<48} PSI={psi:.3f}  {tag}")

    # Split FCU: 20% eval, 80% adapt pool
    np.random.seed(RANDOM_STATE)
    idx_all    = np.random.permutation(len(y_fc))
    n_eval     = int(len(y_fc) * 0.20)
    eval_idx   = idx_all[:n_eval]
    adapt_pool = idx_all[n_eval:]

    X_eval = X_fc_sc[eval_idx]; y_eval = y_fc[eval_idx]

    # Baseline F1
    yp    = clf.predict(X_eval)
    f1_ce = f1_score(y_eval, yp, average='macro', zero_division=0)

    print(f"\nCross-equipment F1: {f1_ce:.4f}")
    print(classification_report(y_eval, yp, zero_division=0))

    # Adaptation
    print("\nAdaptation experiment...")
    print(f"\n  {'%':>6}  {'Records':>10}  {'F1':>8}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")
    print(f"  {'0%':>6}  {'0':>10}  {f1_ce:>8.4f}  {'—':>8}")

    curve = [{'proportion':0.0,'records':0,
              'f1_macro':round(float(f1_ce),4),'delta':0.0}]

    for prop in ADAPT_PROPS[1:]:
        n_adapt = int(len(adapt_pool) * prop)
        if n_adapt == 0: continue
        a_idx  = rng.choice(adapt_pool,
                            min(n_adapt, len(adapt_pool)), replace=False)
        X_a_sc = scaler.transform(X_fc[a_idx])
        y_a    = y_fc[a_idx]

        X_c = np.vstack([X_base_sc, X_a_sc])
        y_c = np.concatenate([y_base, y_a])

        clf_a = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)
        clf_a.fit(X_c, y_c)

        yp_a = clf_a.predict(X_eval)
        f1_a = f1_score(y_eval, yp_a, average='macro', zero_division=0)
        d    = f1_a - f1_ce

        print(f"  {prop:>6.0%}  {n_adapt:>10,}  {f1_a:>8.4f}  {d:>+8.4f}")
        curve.append({'proportion':prop,'records':n_adapt,
                     'f1_macro':round(float(f1_a),4),
                     'delta':round(float(d),4)})

    best = max(curve, key=lambda x: x['f1_macro'])

    print(f"\n{'='*65}")
    print("FINAL RESULTS — COMPARISON ACROSS ALL EQUIPMENT")
    print(f"{'='*65}")
    print(f"  Shared features:              {len(feat_cols)}")
    print(f"  In-sample F1:                 {f1_is:.4f}")
    print()
    print(f"  AHU→AHU (thesis CB):          0.331  (7 shared, distribution shift)")
    print(f"  AHU→AHU (SD-AHU):             0.223  (7 shared, distribution shift)")
    print(f"  AHU→FCU (this experiment):    {f1_ce:.3f}  ({len(feat_cols)} features, 8 shared sensors)")
    print(f"  AHU→RTU:                      0.128  (1 shared, arch. mismatch)")
    print(f"  AHU→Boiler:                   0.096  (1 shared, arch. mismatch)")
    print()
    print(f"  Best FCU after adaptation:    {best['f1_macro']:.4f}"
          f"  (at {best['proportion']:.0%}, {best['records']:,} records)")
    print()

    if f1_ce > 0.25:
        print("  FINDING: FCU shows BETTER transfer than RTU and Boiler.")
        print("  More shared sensors = better cross-equipment generalisation.")
        print("  Thermodynamic similarity (air conditioning) aids transfer.")
    elif f1_ce > 0.15:
        print("  FINDING: FCU partial transfer — better than RTU/Boiler.")
        print("  8 shared sensors help but equipment differences still limit.")
    else:
        print("  FINDING: FCU transfer similar to RTU/Boiler despite more sensors.")
        print("  Equipment architecture dominates over sensor count.")

    os.makedirs('results', exist_ok=True)
    with open('results/fcu_experiment.json', 'w') as f:
        json.dump({
            'experiment'       : 'fcu_cross_equipment',
            'shared_sensors'   : len(SHARED),
            'total_features'   : len(feat_cols),
            'f1_insample'      : round(float(f1_is), 4),
            'f1_cross_equip'   : round(float(f1_ce), 4),
            'best_adapted'     : best['f1_macro'],
            'best_proportion'  : best['proportion'],
            'comparison': {
                'AHU_AHU_thesis': 0.331,
                'AHU_AHU_sdahu' : 0.223,
                'AHU_RTU'       : 0.128,
                'AHU_Boiler'    : 0.096,
                'AHU_FCU'       : round(float(f1_ce), 4),
            },
            'curve': curve,
        }, f, indent=2)
    print("  Saved: results/fcu_experiment.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
