"""
SD-AHU Adaptation Experiment v2
Fixed: exact shared columns, Datetime excluded, trailing spaces stripped
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
SDAHU_DIR    = r'C:\Users\hrslp\Desktop\thesis\data\sdahu'
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

# Exact 7 shared columns confirmed by check_cols.py
SHARED = [
    'AHU: Mixed Air Temperature',
    'AHU: Outdoor Air Temperature',
    'AHU: Return Air Fan Speed Control Signal',
    'AHU: Return Air Temperature',
    'AHU: Supply Air Duct Static Pressure',
    'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Supply Air Temperature',
]

COL_MAP = {
    'SF_CS'      : 'AHU: Supply Air Fan Speed Control Signal',
    'RF_CS'      : 'AHU: Return Air Fan Speed Control Signal',
    'SA_TEMP'    : 'AHU: Supply Air Temperature',
    'SA_SP'      : 'AHU: Supply Air Duct Static Pressure',
    'RA_TEMP'    : 'AHU: Return Air Temperature',
    'OA_TEMP'    : 'AHU: Outdoor Air Temperature',
    'MA_TEMP'    : 'AHU: Mixed Air Temperature',
}

SDAHU_FILES = {
    0: ['AHU_annual.csv'],
    1: ['oa_bias_2_annual.csv', 'oa_bias_-2_annual.csv'],
    2: ['coi_leakage_025_annual.csv'],
}
TRAIN_FILES = {
    'MZVAV-1.csv':1, 'MZVAV-2-1.csv':2, 'SZCAV.csv':3
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean(df):
    df.columns = df.columns.str.strip()   # strip trailing spaces
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df):
    """Engineer features from the 7 shared columns."""
    raw = SHARED
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
    feat['diff_sa_ra'] = df[sa] - df[ra]
    feat['diff_oa_sa'] = df[oa] - df[sa]
    feat['diff_ma_ra'] = df[ma] - df[ra]
    return feat.fillna(0)

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_ashrae():
    frames = []
    for fname, fault_cls in TRAIN_FILES.items():
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df = clean(df)
        if 'Fault Detection Ground Truth' in df.columns:
            label = np.where(df['Fault Detection Ground Truth']==0,
                             0, fault_cls)
        else:
            label = np.zeros(len(df), dtype=int)
        feat = engineer(df)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows "
              f"(H={( label==0).sum():,} F={(label>0).sum():,})")
    return pd.concat(frames, ignore_index=True)

def load_sdahu():
    frames = []
    for label, files in SDAHU_FILES.items():
        for fname in files:
            fpath = os.path.join(SDAHU_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found"); continue
            df = pd.read_csv(fpath, low_memory=False)
            df = df.rename(columns=COL_MAP)
            df = clean(df)
            feat = engineer(df)
            feat['label'] = label
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU ADAPTATION EXPERIMENT v2")
    print("7 shared columns · clean data · no zero-fill")
    print("=" * 65)

    print(f"\nUsing {len(SHARED)} shared columns:")
    for c in SHARED: print(f"  ✓ {c}")

    print("\nLoading ASHRAE training data...")
    ashrae = load_ashrae()
    print("\nLoading SD-AHU data...")
    sdahu  = load_sdahu()

    feat_cols = [c for c in ashrae.columns if c != 'label']
    print(f"\nTotal features engineered: {len(feat_cols)}")

    # ── Train base model (ASHRAE only) ────────────────────────────────────────
    print("\n" + "="*65)
    print("BASE MODEL — ASHRAE only (no SD-AHU data)")
    print("="*65)
    y_tr = ashrae['label'].values
    X_tr = ashrae[feat_cols]

    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr):
        idx = np.where(y_tr==cls)[0]
        n   = min(len(idx), 30000)
        ch  = rng.choice(idx, n, replace=False)
        sub_X.append(X_tr.iloc[ch])
        sub_y.append(y_tr[ch])
    X_base = pd.concat(sub_X)
    y_base = np.concatenate(sub_y)

    scaler = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)

    clf_base = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    clf_base.fit(X_base_sc, y_base)

    in_sample_f1 = f1_score(clf_base.predict(X_base_sc), y_base,
                             average='macro', zero_division=0)
    print(f"  In-sample F1: {in_sample_f1:.4f}")

    # ── Split SD-AHU: fixed eval 20% + adapt pool 80% ────────────────────────
    y_sd = sdahu['label'].values
    X_sd = sdahu[feat_cols].values

    np.random.seed(RANDOM_STATE)
    idx_all = np.arange(len(y_sd))
    np.random.shuffle(idx_all)
    n_eval     = int(len(y_sd) * 0.20)
    eval_idx   = idx_all[:n_eval]
    adapt_pool = idx_all[n_eval:]

    X_eval_sc = scaler.transform(X_sd[eval_idx])
    y_eval    = y_sd[eval_idx]

    mask3     = np.isin(y_eval, [0,1,2])
    X_eval3   = X_eval_sc[mask3]
    y_eval3   = y_eval[mask3]

    # Baseline (0% adaptation)
    yp_base   = clf_base.predict(X_eval3)
    f1_base   = f1_score(y_eval3, yp_base, average='macro', zero_division=0)
    rep_base  = classification_report(y_eval3, yp_base,
                    labels=[0,1,2], zero_division=0, output_dict=True)
    f1a_base  = rep_base.get('1',{}).get('f1-score',0)
    f1b_base  = rep_base.get('2',{}).get('f1-score',0)

    print(f"\n  Baseline CB F1 (0% SD-AHU data): {f1_base:.4f}")
    print(f"    Fault-A F1: {f1a_base:.4f}  |  Fault-B F1: {f1b_base:.4f}")
    print(f"  (Thesis equivalent CB F1:        0.331)")

    # ── Adaptation loop ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("ADAPTATION RESULTS")
    print("="*65)
    print(f"\n  {'%':>6}  {'Records':>10}  {'Macro F1':>9}  "
          f"{'Δ vs 0%':>9}  {'Fault-A':>8}  {'Fault-B':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")
    print(f"  {'0%':>6}  {'0':>10}  {f1_base:>9.4f}  "
          f"{'—':>9}  {f1a_base:>8.4f}  {f1b_base:>8.4f}")

    results_list = [{'proportion':0.0,'records':0,
                     'f1_macro':round(float(f1_base),4),
                     'f1_fault_a':round(float(f1a_base),4),
                     'f1_fault_b':round(float(f1b_base),4),
                     'delta':0.0}]

    for prop in ADAPT_PROPS[1:]:
        n_adapt  = int(len(adapt_pool) * prop)
        if n_adapt == 0: continue
        a_idx    = rng.choice(adapt_pool,
                              min(n_adapt, len(adapt_pool)), replace=False)
        X_adapt  = scaler.transform(X_sd[a_idx])
        y_adapt  = y_sd[a_idx]

        X_comb   = np.vstack([X_base_sc, X_adapt])
        y_comb   = np.concatenate([y_base, y_adapt])

        clf_a = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)
        clf_a.fit(X_comb, y_comb)

        yp    = clf_a.predict(X_eval3)
        f1    = f1_score(y_eval3, yp, average='macro', zero_division=0)
        rep   = classification_report(y_eval3, yp,
                    labels=[0,1,2], zero_division=0, output_dict=True)
        f1a   = rep.get('1',{}).get('f1-score',0)
        f1b   = rep.get('2',{}).get('f1-score',0)
        delta = f1 - f1_base

        print(f"  {prop:>6.0%}  {n_adapt:>10,}  {f1:>9.4f}  "
              f"{delta:>+9.4f}  {f1a:>8.4f}  {f1b:>8.4f}")

        results_list.append({'proportion':prop,'records':n_adapt,
                             'f1_macro':round(float(f1),4),
                             'f1_fault_a':round(float(f1a),4),
                             'f1_fault_b':round(float(f1b),4),
                             'delta':round(float(delta),4)})

    best = max(results_list, key=lambda x: x['f1_macro'])

    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"  Baseline CB F1:                 {f1_base:.4f}")
    print(f"  Best adapted CB F1:             {best['f1_macro']:.4f}  "
          f"(at {best['proportion']:.0%}, {best['records']:,} records)")
    print(f"  Fault-B at baseline:            {f1b_base:.4f}")
    print(f"  Fault-B at best adaptation:     {best['f1_fault_b']:.4f}")
    print()
    print(f"  Thesis comparison:")
    print(f"    Thesis baseline CB F1:        0.331")
    print(f"    Thesis adapted F1 (10%):      0.991")
    print(f"    SD-AHU baseline CB F1:        {f1_base:.4f}")
    print(f"    SD-AHU best adapted F1:       {best['f1_macro']:.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/sdahu_adaptation_v2.json','w') as f:
        json.dump({'experiment':'sdahu_adaptation_v2',
                   'shared_features':len(feat_cols),
                   'baseline_f1':round(float(f1_base),4),
                   'best_f1':best['f1_macro'],
                   'best_proportion':best['proportion'],
                   'best_records':best['records'],
                   'thesis_baseline':0.331,
                   'thesis_adapted':0.991,
                   'curve':results_list}, f, indent=2)
    print("\n  Saved: results/sdahu_adaptation_v2.json")
    print("="*65)

if __name__ == '__main__':
    main()
