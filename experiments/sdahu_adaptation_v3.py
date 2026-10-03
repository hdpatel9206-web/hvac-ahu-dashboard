"""
SD-AHU Adaptation Experiment v3
Correct shared columns across ALL three ASHRAE files + SD-AHU
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASHRAE_DIR   = os.path.join(REPO_ROOT, 'data')
SDAHU_DIR    = os.path.join(REPO_ROOT, 'data', 'sdahu')
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

# True intersection across MZVAV-1, MZVAV-2-1, SZCAV and SD-AHU
# SZCAV lacks: Return Air Fan Speed Control Signal, Supply Air Duct Static Pressure
SHARED = [
    'AHU: Mixed Air Temperature',
    'AHU: Outdoor Air Temperature',
    'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Supply Air Temperature',
]

COL_MAP = {
    'SF_CS'  : 'AHU: Supply Air Fan Speed Control Signal',
    'RF_CS'  : 'AHU: Return Air Fan Speed Control Signal',
    'SA_TEMP': 'AHU: Supply Air Temperature',
    'SA_SP'  : 'AHU: Supply Air Duct Static Pressure',
    'RA_TEMP': 'AHU: Return Air Temperature',
    'OA_TEMP': 'AHU: Outdoor Air Temperature',
    'MA_TEMP': 'AHU: Mixed Air Temperature',
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
    df.columns = df.columns.str.strip()
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df):
    # Use only shared cols that exist in this file
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

def align(feat, feat_cols):
    """Align feature df to master feature column list."""
    out = pd.DataFrame(0.0, index=feat.index, columns=feat_cols)
    for c in feat_cols:
        if c in feat.columns:
            out[c] = feat[c].values
    return out

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_ashrae():
    frames = []
    for fname, fault_cls in TRAIN_FILES.items():
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df = clean(df)
        label = np.where(
            df['Fault Detection Ground Truth']==0, 0, fault_cls
        ) if 'Fault Detection Ground Truth' in df.columns \
          else np.zeros(len(df), int)
        feat = engineer(df)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows "
              f"(H={(label==0).sum():,} F={(label>0).sum():,})")
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
    print("SD-AHU ADAPTATION EXPERIMENT v3")
    print("True shared columns · all 3 ASHRAE files + SD-AHU")
    print("=" * 65)

    print(f"\nShared columns ({len(SHARED)}):")
    for c in SHARED: print(f"  ✓ {c}")

    print("\nLoading ASHRAE training data...")
    ashrae = load_ashrae()
    print("\nLoading SD-AHU data...")
    sdahu  = load_sdahu()

    # Build unified feature column list
    feat_cols = sorted(set(ashrae.columns) | set(sdahu.columns) - {'label'})
    feat_cols = [c for c in feat_cols if c != 'label']

    # Align both datasets to same feature space
    print(f"\nFeatures after alignment: {len(feat_cols)}")
    y_tr    = ashrae['label'].values
    X_tr_df = align(ashrae, feat_cols)
    y_sd    = sdahu['label'].values
    X_sd_df = align(sdahu,  feat_cols)

    # Subsample ASHRAE for balance
    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr):
        idx = np.where(y_tr==cls)[0]
        n   = min(len(idx), 30000)
        ch  = rng.choice(idx, n, replace=False)
        sub_X.append(X_tr_df.iloc[ch])
        sub_y.append(y_tr[ch])
    X_base = pd.concat(sub_X).values
    y_base = np.concatenate(sub_y)

    print(f"\nTraining base model...")
    print(f"  Training rows: {len(y_base):,}  "
          f"classes: {np.unique(y_base)}")
    scaler    = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)

    clf_base = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    clf_base.fit(X_base_sc, y_base)
    f1_is = f1_score(clf_base.predict(X_base_sc), y_base,
                     average='macro', zero_division=0)
    print(f"  In-sample F1: {f1_is:.4f}")

    # Split SD-AHU: 20% fixed eval, 80% adapt pool
    X_sd = X_sd_df.values
    idx_all = np.random.default_rng(RANDOM_STATE).permutation(len(y_sd))
    n_eval     = int(len(y_sd) * 0.20)
    eval_idx   = idx_all[:n_eval]
    adapt_pool = idx_all[n_eval:]

    X_eval_sc = scaler.transform(X_sd[eval_idx])
    y_eval    = y_sd[eval_idx]
    mask3     = np.isin(y_eval, [0,1,2])
    X_eval3   = X_eval_sc[mask3]
    y_eval3   = y_eval[mask3]

    # Baseline
    yp0   = clf_base.predict(X_eval3)
    f1_0  = f1_score(y_eval3, yp0, average='macro', zero_division=0)
    rep0  = classification_report(y_eval3, yp0,
                labels=[0,1,2], zero_division=0, output_dict=True)
    f1a_0 = rep0.get('1',{}).get('f1-score',0)
    f1b_0 = rep0.get('2',{}).get('f1-score',0)

    print(f"\n{'='*65}")
    print(f"ADAPTATION RESULTS")
    print(f"{'='*65}")
    print(f"\n  {'%':>6}  {'Records':>10}  {'Macro F1':>9}  "
          f"{'Δ':>7}  {'Fault-A':>8}  {'Fault-B':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*8}")
    print(f"  {'0%':>6}  {'0':>10}  {f1_0:>9.4f}  "
          f"{'—':>7}  {f1a_0:>8.4f}  {f1b_0:>8.4f}")

    curve = [{'proportion':0.0,'records':0,
              'f1_macro':round(float(f1_0),4),
              'f1_fault_a':round(float(f1a_0),4),
              'f1_fault_b':round(float(f1b_0),4),
              'delta':0.0}]

    for prop in ADAPT_PROPS[1:]:
        n_adapt = int(len(adapt_pool) * prop)
        if n_adapt == 0: continue
        a_idx   = rng.choice(adapt_pool,
                             min(n_adapt,len(adapt_pool)), replace=False)
        X_a_sc  = scaler.transform(X_sd[a_idx])
        y_a     = y_sd[a_idx]

        X_c = np.vstack([X_base_sc, X_a_sc])
        y_c = np.concatenate([y_base, y_a])

        clf_a = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)
        clf_a.fit(X_c, y_c)

        yp  = clf_a.predict(X_eval3)
        f1  = f1_score(y_eval3, yp, average='macro', zero_division=0)
        rep = classification_report(y_eval3, yp,
                labels=[0,1,2], zero_division=0, output_dict=True)
        f1a = rep.get('1',{}).get('f1-score',0)
        f1b = rep.get('2',{}).get('f1-score',0)
        d   = f1 - f1_0

        print(f"  {prop:>6.0%}  {n_adapt:>10,}  {f1:>9.4f}  "
              f"{d:>+7.4f}  {f1a:>8.4f}  {f1b:>8.4f}")
        curve.append({'proportion':prop,'records':n_adapt,
                      'f1_macro':round(float(f1),4),
                      'f1_fault_a':round(float(f1a),4),
                      'f1_fault_b':round(float(f1b),4),
                      'delta':round(float(d),4)})

    best = max(curve, key=lambda x: x['f1_macro'])

    print(f"\n{'='*65}")
    print(f"FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  Shared features used:           {len(feat_cols)}")
    print(f"  In-sample F1:                   {f1_is:.4f}")
    print(f"  Baseline CB F1 (0% adapt):      {f1_0:.4f}")
    print(f"  Best adapted CB F1:             {best['f1_macro']:.4f}"
          f"  (at {best['proportion']:.0%}, {best['records']:,} records)")
    print(f"  Fault-B at baseline:            {f1b_0:.4f}")
    print(f"  Fault-B at best:                {best['f1_fault_b']:.4f}")
    print()
    print(f"  Thesis comparison:")
    print(f"    Thesis CB baseline:           0.331")
    print(f"    Thesis adapted (10%):         0.991")
    print(f"    SD-AHU CB baseline:           {f1_0:.4f}")
    print(f"    SD-AHU best adapted:          {best['f1_macro']:.4f}")

    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    with open(os.path.join(REPO_ROOT, 'results', 'sdahu_adaptation_v3.json'),'w') as f:
        json.dump({'experiment':'sdahu_adaptation_v3',
                   'shared_cols':SHARED,
                   'n_features':len(feat_cols),
                   'f1_insample':round(float(f1_is),4),
                   'f1_baseline':round(float(f1_0),4),
                   'f1_fault_a_baseline':round(float(f1a_0),4),
                   'f1_fault_b_baseline':round(float(f1b_0),4),
                   'best_f1':best['f1_macro'],
                   'best_proportion':best['proportion'],
                   'best_records':best['records'],
                   'thesis_baseline':0.331,
                   'thesis_adapted':0.991,
                   'curve':curve}, f, indent=2)
    print("\n  Saved: results/sdahu_adaptation_v3.json")
    print("="*65)

if __name__ == '__main__':
    main()
