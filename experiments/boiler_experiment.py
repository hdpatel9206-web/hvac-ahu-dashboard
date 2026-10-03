"""
Boiler Plant Cross-Equipment Experiment
AHU-trained model tested on Boiler Plant
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASHRAE_DIR  = os.path.join(REPO_ROOT, 'data')
BOILER_DIR  = os.path.join(REPO_ROOT, 'data', 'boiler')
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]

# Only OA_TEMP maps across both systems
COL_MAP = {
    'OA_TEMP': 'AHU: Outdoor Air Temperature',
}

# Use all boiler files grouped by fault type
BOILER_FILES = {
    0: ['BoilerPlant.csv'],
    1: ['BoilerPlant_boiler_bias_2.csv', 'BoilerPlant_boiler_bias_-2.csv'],
    2: ['BoilerPlant_boiler_foul_065.csv', 'BoilerPlant_boiler_foul_080.csv'],
    3: ['BoilerPlant_boiler_PI.csv'],
    4: ['BoilerPlant_hot_water_pressure_bias_10.csv',
        'BoilerPlant_hot_water_pressure_bias_-10.csv'],
    5: ['BoilerPlant_hot_water_temp_bias_2.csv',
        'BoilerPlant_hot_water_temp_bias_-2.csv'],
}

TRAIN_FILES = {
    'MZVAV-1.csv': 1, 'MZVAV-2-1.csv': 2, 'SZCAV.csv': 3
}

SHARED = ['AHU: Outdoor Air Temperature']

# ── HELPERS ──────────────────────────────────────────────────────────────────

def clean(df):
    df.columns = df.columns.str.strip()
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df, cols):
    raw = [c for c in cols if c in df.columns]
    if not raw:
        return pd.DataFrame(index=df.index)
    data = df[raw].copy().fillna(0)
    frames = [data]
    for w in WINDOW_SIZES:
        rm = data.rolling(w, min_periods=1).mean()
        rs = data.rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    return pd.concat(frames, axis=1).fillna(0)

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
            df.get('Fault Detection Ground Truth', pd.Series(0, index=df.index)) == 0,
            0, fcls)
        feat = engineer(df, SHARED)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows")
    return pd.concat(frames, ignore_index=True)

def load_boiler():
    frames = []
    for label, files in BOILER_FILES.items():
        for fname in files:
            fpath = os.path.join(BOILER_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found"); continue
            df = clean(pd.read_csv(fpath, low_memory=False))
            df = df.rename(columns=COL_MAP)
            feat = engineer(df, SHARED)
            feat['label'] = label
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("BOILER PLANT CROSS-EQUIPMENT EXPERIMENT")
    print("AHU-trained model vs Boiler Plant")
    print("=" * 65)

    print(f"\nShared columns: {SHARED}")
    print("Note: Only OA_TEMP shared. This tests architectural mismatch.")

    print("\nLoading ASHRAE training data...")
    ashrae = load_ashrae()
    print("\nLoading Boiler data...")
    boiler = load_boiler()

    feat_cols = [c for c in ashrae.columns if c != 'label']
    print(f"\nFeatures: {len(feat_cols)}")
    print(f"ASHRAE rows: {len(ashrae):,}")
    print(f"Boiler rows: {len(boiler):,}")

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

    # Boiler evaluation
    y_bo = boiler['label'].values
    X_bo = boiler[feat_cols].values
    X_bo_sc = scaler.transform(X_bo)

    # PSI on OA_TEMP only
    print("\nPSI Analysis:")
    for i, col in enumerate(feat_cols):
        if 'rm' not in col and 'rs' not in col:
            psi = compute_psi(X_base_sc[:, i], X_bo_sc[:, i])
            tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
            print(f"  {col:<45} PSI={psi:.3f}  {tag}")

    # Cross-equipment F1
    print("\nCross-equipment evaluation...")
    np.random.seed(RANDOM_STATE)
    idx_all = np.random.permutation(len(y_bo))
    n_eval  = int(len(y_bo) * 0.20)
    eval_idx = idx_all[:n_eval]
    adapt_pool = idx_all[n_eval:]

    X_eval = X_bo_sc[eval_idx]; y_eval = y_bo[eval_idx]
    yp = clf.predict(X_eval)
    f1_ce = f1_score(y_eval, yp, average='macro', zero_division=0)

    print(f"\n  Cross-equipment F1: {f1_ce:.4f}")
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
        X_a_sc = scaler.transform(X_bo[a_idx])
        y_a    = y_bo[a_idx]

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
    print("FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Shared features:              {len(feat_cols)} (OA temp only)")
    print(f"  In-sample F1:                 {f1_is:.4f}")
    print(f"  Thesis AHU\u2192AHU CB F1:        0.331")
    print(f"  SD-AHU CB F1:                 0.223")
    print(f"  RTU cross-equip F1:           0.128")
    print(f"  Boiler cross-equip F1:        {f1_ce:.4f}")
    print(f"  Best after adaptation:        {best['f1_macro']:.4f}"
          f"  (at {best['proportion']:.0%})")
    print()

    if f1_ce < 0.20:
        print("  FINDING: Confirms architectural mismatch pattern.")
        print("  Boiler shares no meaningful sensors with AHU.")
        print("  Same pattern as RTU (F1=0.128). Equipment gap > building gap.")
    elif f1_ce > 0.30:
        print("  FINDING: Unexpected. Boiler partially transfers from AHU.")
        print("  OA temperature may carry enough fault signal.")

    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    with open(os.path.join(REPO_ROOT, 'results', 'boiler_experiment.json'), 'w') as f:
        json.dump({
            'experiment'      : 'boiler_cross_equipment',
            'shared_features' : feat_cols,
            'f1_insample'     : round(float(f1_is), 4),
            'f1_cross_equip'  : round(float(f1_ce), 4),
            'thesis_ahu_cb'   : 0.331,
            'sdahu_cb'        : 0.223,
            'rtu_cross_equip' : 0.128,
            'best_adapted'    : best['f1_macro'],
            'curve'           : curve,
        }, f, indent=2)
    print("\n  Saved: results/boiler_experiment.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
