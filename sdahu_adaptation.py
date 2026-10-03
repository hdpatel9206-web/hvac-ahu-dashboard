"""
SD-AHU Adaptation Experiment
Tests how much target-building data is needed to recover performance
Mirrors the thesis fine-tuning experiment exactly
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
ASHRAE_DIR   = r'C:\Users\hrslp\Desktop\thesis\data'
SDAHU_DIR    = r'C:\Users\hrslp\Desktop\thesis\data\sdahu'
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]

# Adaptation proportions to test (mirrors thesis learning curve)
ADAPT_PROPORTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

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
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer(df, shared_cols):
    raw = [c for c in shared_cols if c in df.columns]
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

def load_ashrae(shared):
    fault_map = {'MZVAV-1.csv':1,'MZVAV-2-1.csv':2,'SZCAV.csv':3}
    frames = []
    for fname in TRAIN_FILES:
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df = clean_df(df)
        if 'Fault Detection Ground Truth' in df.columns:
            df['label'] = np.where(
                df['Fault Detection Ground Truth']==0, 0,
                fault_map.get(fname,1))
        else:
            df['label'] = 0
        lbl = df.pop('label')
        feat = engineer(df[shared].copy(), shared)
        feat['label'] = lbl.values
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows")
    return pd.concat(frames, ignore_index=True)

def load_sdahu(shared):
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
            df = clean_df(df)
            lbl_val = label
            feat = engineer(df[shared].copy(), shared)
            feat['label'] = lbl_val
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU ADAPTATION EXPERIMENT")
    print("How much target-building data recovers performance?")
    print("=" * 65)

    # Determine shared columns
    sample_sdahu = pd.read_csv(
        os.path.join(SDAHU_DIR,'AHU_annual.csv'), nrows=5
    ).rename(columns=COL_MAP)
    sample_ashrae = clean_df(pd.read_csv(
        os.path.join(ASHRAE_DIR,'MZVAV-1.csv'), nrows=5
    ))
    shared = sorted(set(sample_sdahu.columns) &
                    set(sample_ashrae.columns) -
                    {'Fault Detection Ground Truth'})
    print(f"\nShared sensor columns: {len(shared)}")
    for c in shared:
        print(f"  ✓ {c}")

    # Load data
    print("\nLoading ASHRAE training data...")
    ashrae = load_ashrae(shared)
    print("\nLoading SD-AHU data...")
    sdahu  = load_sdahu(shared)

    feat_cols = [c for c in ashrae.columns if c != 'label']
    print(f"\nFeatures: {len(feat_cols)}")

    # Train base model on ASHRAE only
    print("\n" + "="*65)
    print("STEP A — Base model (ASHRAE only, no SD-AHU data)")
    print("="*65)
    y_tr = ashrae['label'].values
    X_tr = ashrae[feat_cols]

    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr):
        idx = np.where(y_tr==cls)[0]
        n   = min(len(idx), 30000)
        sub_X.append(X_tr.iloc[rng.choice(idx,n,replace=False)])
        sub_y.append(y_tr[rng.choice(idx,n,replace=False)])
    X_base = pd.concat(sub_X)
    y_base = np.concatenate(sub_y)

    scaler = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)

    clf_base = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf_base.fit(X_base_sc, y_base)

    # Split SD-AHU into adapt pool and fixed eval set
    # Use 80% for potential adaptation, 20% fixed eval (never used in training)
    y_sd = sdahu['label'].values
    X_sd = sdahu[feat_cols]

    np.random.seed(RANDOM_STATE)
    n_sd = len(y_sd)
    all_idx = np.arange(n_sd)
    np.random.shuffle(all_idx)
    eval_idx  = all_idx[:int(n_sd * 0.20)]   # fixed 20% eval set
    adapt_pool= all_idx[int(n_sd * 0.20):]   # 80% available for adaptation

    X_eval = scaler.transform(X_sd.iloc[eval_idx])
    y_eval = y_sd[eval_idx]

    # Baseline: 0% adaptation
    mask3 = np.isin(y_eval, [0,1,2])
    yp0   = clf_base.predict(X_eval[mask3])
    f1_0  = f1_score(y_eval[mask3], yp0, average='macro', zero_division=0)
    print(f"\n  Baseline CB F1 (0% adaptation): {f1_0:.4f}")
    print(f"  (Thesis equivalent: 0.331)")

    # Adaptation loop
    print("\n" + "="*65)
    print("STEP B — Adaptation loop")
    print("="*65)
    print(f"\n  {'Proportion':>12}  {'Records':>10}  {'F1':>7}  {'vs baseline':>12}  {'Fault-A':>8}  {'Fault-B':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*8}")

    results_list = [{'proportion':0.0, 'records':0,
                     'f1_macro':round(float(f1_0),4),
                     'delta':0.0}]

    for prop in ADAPT_PROPORTIONS[1:]:
        n_adapt = int(len(adapt_pool) * prop)
        if n_adapt == 0:
            continue

        # Sample adaptation data from pool
        adapt_idx = rng.choice(adapt_pool, min(n_adapt, len(adapt_pool)),
                               replace=False)
        X_adapt_raw = X_sd.iloc[adapt_idx]
        y_adapt     = y_sd[adapt_idx]

        # Combine ASHRAE + SD-AHU adaptation data
        X_adapt_sc  = scaler.transform(X_adapt_raw)
        X_combined  = np.vstack([X_base_sc, X_adapt_sc])
        y_combined  = np.concatenate([y_base, y_adapt])

        # Retrain
        clf_adapt = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf_adapt.fit(X_combined, y_combined)

        # Evaluate on fixed eval set
        yp = clf_adapt.predict(X_eval[mask3])
        f1 = f1_score(y_eval[mask3], yp, average='macro', zero_division=0)

        # Per-class F1
        rep = classification_report(
            y_eval[mask3], yp,
            labels=[0,1,2], zero_division=0, output_dict=True)
        f1_a = rep.get('1',{}).get('f1-score',0)
        f1_b = rep.get('2',{}).get('f1-score',0)

        delta = f1 - f1_0
        print(f"  {prop:>12.0%}  {n_adapt:>10,}  {f1:>7.4f}  "
              f"{delta:>+12.4f}  {f1_a:>8.4f}  {f1_b:>8.4f}")

        results_list.append({
            'proportion'  : prop,
            'records'     : n_adapt,
            'f1_macro'    : round(float(f1),  4),
            'f1_fault_a'  : round(float(f1_a),4),
            'f1_fault_b'  : round(float(f1_b),4),
            'delta'       : round(float(delta),4),
        })

    # Find recovery threshold
    best = max(results_list, key=lambda x: x['f1_macro'])
    thresh = next((r for r in results_list if r['f1_macro'] >= 0.6), None)

    print("\n" + "="*65)
    print("ADAPTATION RESULTS SUMMARY")
    print("="*65)
    print(f"  Baseline CB F1 (0% adaptation):           {f1_0:.4f}")
    print(f"  Best CB F1 achieved:                      {best['f1_macro']:.4f}")
    print(f"  Records needed for best:                  {best['records']:,}")
    print(f"  Proportion needed for best:               {best['proportion']:.0%}")
    if thresh:
        print(f"\n  F1 > 0.60 first achieved at:              {thresh['proportion']:.0%}")
        print(f"  Records at that threshold:                {thresh['records']:,}")
    print()
    print("  Thesis comparison:")
    print(f"    Thesis baseline CB F1:                  0.331")
    print(f"    Thesis adapted F1 at 10%:               0.991")
    print(f"    SD-AHU baseline CB F1:                  {f1_0:.4f}")
    print(f"    SD-AHU best adapted F1:                 {best['f1_macro']:.4f}")

    os.makedirs('results', exist_ok=True)
    out = {
        'experiment'     : 'sdahu_adaptation',
        'baseline_f1'    : round(float(f1_0), 4),
        'best_f1'        : best['f1_macro'],
        'best_proportion': best['proportion'],
        'best_records'   : best['records'],
        'thesis_baseline': 0.331,
        'thesis_adapted' : 0.991,
        'adaptation_curve': results_list,
    }
    with open('results/sdahu_adaptation.json','w') as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: results/sdahu_adaptation.json")
    print("="*65)

if __name__ == '__main__':
    main()
