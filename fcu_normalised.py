"""
FCU Normalised Experiment
Fix for Category 2 (thermodynamically similar equipment)
Physics-based min-max normalisation per sensor role
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
FCU_DIR      = r'C:\Users\hrslp\Desktop\thesis\data\fcu'
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.02, 0.05, 0.10]

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

# Physics-based operating ranges per sensor role
# These are the universal physical bounds regardless of equipment size
# Source: ASHRAE Handbook + engineering judgement
PHYSICS_RANGES = {
    'AHU: Cooling Coil Valve Control Signal'  : (0.0,  1.0),
    'AHU: Heating Coil Valve Control Signal'  : (0.0,  1.0),
    'AHU: Mixed Air Temperature'              : (-10.0, 50.0),
    'AHU: Outdoor Air Damper Control Signal'  : (0.0,  1.0),
    'AHU: Outdoor Air Temperature'            : (-20.0, 45.0),
    'AHU: Return Air Temperature'             : (10.0,  35.0),
    'AHU: Supply Air Fan Speed Control Signal': (0.0,  1.0),
    'AHU: Supply Air Temperature'             : (5.0,  35.0),
}

FCU_FILES = {
    0: ['FCU_FaultFree.csv'],
    1: ['FCU_Control_CoolingReverse.csv',
        'FCU_Control_HeatingReverse.csv',
        'FCU_Control_Unstable.csv'],
    2: ['FCU_FilterRestriction_10%.csv',
        'FCU_FilterRestriction_20%.csv',
        'FCU_FilterRestriction_50%.csv'],
    3: ['FCU_Fouling_Cooling_Airside_Minor.csv',
        'FCU_Fouling_Cooling_Waterside_Minor.csv'],
    4: ['FCU_Fouling_Heating_Airside_Minor.csv',
        'FCU_Fouling_Heating_Waterside_Minor.csv'],
    5: ['FCU_OABlockage.csv',
        'FCU_OADMPRLeak_20.csv',
        'FCU_OADMPRStuck_0.csv'],
    6: ['FCU_SensorBias_RMTemp_+2C.csv',
        'FCU_SensorBias_RMTemp_-2C.csv'],
    7: ['FCU_VLVLeak_Cooling_20.csv',
        'FCU_VLVStuck_Cooling_0.csv'],
    8: ['FCU_VLVLeak_Heating_20.csv',
        'FCU_VLVStuck_Heating_0.csv'],
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

def physics_normalise(df):
    """Normalise each shared sensor by its universal physical range."""
    df_out = df.copy()
    for col, (lo, hi) in PHYSICS_RANGES.items():
        if col in df_out.columns:
            df_out[col] = (df_out[col] - lo) / (hi - lo)
            # Clip to [-0.1, 1.1] to allow slight out-of-range values
            df_out[col] = df_out[col].clip(-0.1, 1.1)
    return df_out

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
    lo = min(ref.min(), tgt.min()); hi = max(ref.max(), tgt.max())
    if lo == hi: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    rp = np.histogram(ref, bins=bins)[0].astype(float) + eps
    tp = np.histogram(tgt, bins=bins)[0].astype(float) + eps
    rp /= rp.sum(); tp /= tp.sum()
    return float(np.sum((tp - rp) * np.log(tp / rp)))

def load_ashrae(normalise=False):
    frames = []
    for fname, fcls in TRAIN_FILES.items():
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath): continue
        df = clean(pd.read_csv(fpath, low_memory=False))
        if normalise:
            df = physics_normalise(df)
        label = np.where(
            df.get('Fault Detection Ground Truth',
                   pd.Series(0, index=df.index)) == 0, 0, fcls)
        feat = engineer(df)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows")
    return pd.concat(frames, ignore_index=True)

def load_fcu(normalise=False):
    frames = []
    loaded = 0
    for label, files in FCU_FILES.items():
        for fname in files:
            fpath = os.path.join(FCU_DIR, fname)
            if not os.path.exists(fpath): continue
            df = clean(pd.read_csv(fpath, low_memory=False))
            df = df.rename(columns=COL_MAP)
            if normalise:
                df = physics_normalise(df)
            feat = engineer(df)
            feat['label'] = label
            frames.append(feat)
            loaded += 1
    print(f"  Loaded {loaded} FCU files")
    return pd.concat(frames, ignore_index=True)

def run_experiment(normalise, label):
    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*65}")

    ashrae = load_ashrae(normalise)
    fcu    = load_fcu(normalise)

    feat_cols = [c for c in ashrae.columns if c != 'label']

    # Train
    y_tr = ashrae['label'].values
    X_tr = ashrae[feat_cols].values
    rng  = np.random.default_rng(RANDOM_STATE)
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

    # PSI after normalisation
    y_fc    = fcu['label'].values
    X_fc    = fcu[feat_cols].values
    X_fc_sc = scaler.transform(X_fc)

    print(f"\nPSI (after {label}):")
    raw_cols = [c for c in feat_cols
                if '_rm' not in c and '_rs' not in c and 'diff' not in c]
    psi_vals = {}
    for i, col in enumerate(feat_cols):
        if col in raw_cols:
            psi = compute_psi(X_base_sc[:, i], X_fc_sc[:, i])
            psi_vals[col] = psi
            tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
            print(f"  {col[:48]:<48} PSI={psi:.3f}  {tag}")

    # Eval
    np.random.seed(RANDOM_STATE)
    idx_all  = np.random.permutation(len(y_fc))
    eval_idx = idx_all[:15000]
    adapt_pool = idx_all[15000:]

    X_eval = X_fc_sc[eval_idx]; y_eval = y_fc[eval_idx]
    yp     = clf.predict(X_eval)
    f1_0   = f1_score(y_eval, yp, average='macro', zero_division=0)
    print(f"\nBaseline F1: {f1_0:.4f}  (original: 0.0211)")

    # Adaptation
    print(f"\n  {'%':>6}  {'Records':>10}  {'F1':>8}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")
    print(f"  {'0%':>6}  {'0':>10}  {f1_0:>8.4f}  {'—':>8}")

    curve = [{'proportion':0.0,'records':0,'f1':round(float(f1_0),4)}]

    for prop in ADAPT_PROPS[1:]:
        n_a   = min(int(len(adapt_pool)*prop), 50000)
        a_idx = rng.choice(adapt_pool, n_a, replace=False)
        X_a   = scaler.transform(X_fc[a_idx])
        y_a   = y_fc[a_idx]
        X_c   = np.vstack([X_base_sc, X_a])
        y_c   = np.concatenate([y_base, y_a])
        clf_a = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)
        clf_a.fit(X_c, y_c)
        yp_a = clf_a.predict(X_eval)
        f1_a = f1_score(y_eval, yp_a, average='macro', zero_division=0)
        d    = f1_a - f1_0
        print(f"  {prop:>6.0%}  {n_a:>10,}  {f1_a:>8.4f}  {d:>+8.4f}")
        curve.append({'proportion':prop,'records':n_a,'f1':round(float(f1_a),4)})

    return {
        'label'        : label,
        'normalised'   : normalise,
        'baseline_f1'  : round(float(f1_0), 4),
        'psi_after'    : {k: round(v,3) for k,v in psi_vals.items()},
        'curve'        : curve
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FCU NORMALISATION FIX EXPERIMENT")
    print("Comparing standard scaling vs physics-based normalisation")
    print("=" * 65)

    print("\nLoading ASHRAE data...")
    print("\nRun 1 — Standard scaling (original approach):")
    r1 = run_experiment(normalise=False, label="Standard scaling (original)")

    print("\n\nRun 2 — Physics-based normalisation (fix):")
    r2 = run_experiment(normalise=True,  label="Physics-based normalisation")

    print(f"\n{'='*65}")
    print("COMPARISON: Standard vs Physics Normalisation")
    print(f"{'='*65}")
    print(f"  {'Metric':<40} {'Standard':>10}  {'Physics Norm':>13}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*13}")
    print(f"  {'Baseline F1':<40} {r1['baseline_f1']:>10.4f}  {r2['baseline_f1']:>13.4f}")

    for p in ADAPT_PROPS[1:]:
        v1 = next((x['f1'] for x in r1['curve'] if x['proportion']==p), 0)
        v2 = next((x['f1'] for x in r2['curve'] if x['proportion']==p), 0)
        print(f"  {f'After {p:.0%} adaptation':<40} {v1:>10.4f}  {v2:>13.4f}")

    improvement = r2['baseline_f1'] - r1['baseline_f1']
    print(f"\n  Baseline improvement from normalisation: {improvement:+.4f}")

    if improvement > 0.05:
        finding = "SIGNIFICANT — physics normalisation reduces cross-equipment gap"
    elif improvement > 0.01:
        finding = "MODEST — normalisation helps but not sufficient alone"
    else:
        finding = "MINIMAL — operating range difference is not the primary barrier"
    print(f"  Finding: {finding}")

    print(f"\n  PSI comparison (fan speed):")
    fs = 'AHU: Supply Air Fan Speed Control Signal'
    print(f"    Standard:   PSI = {r1['psi_after'].get(fs, 'N/A')}")
    print(f"    Normalised: PSI = {r2['psi_after'].get(fs, 'N/A')}")

    os.makedirs('results', exist_ok=True)
    with open('results/fcu_normalisation_fix.json', 'w') as f:
        json.dump({'standard': r1, 'normalised': r2,
                   'improvement': round(float(improvement), 4),
                   'finding': finding}, f, indent=2)
    print("\n  Saved: results/fcu_normalisation_fix.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
