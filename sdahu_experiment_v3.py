"""
SD-AHU Cross-Building Experiment v3
Official column mapping from PDF Table 2
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import joblib
warnings.filterwarnings('ignore')

MODEL_PATH   = 'models/rf_model.pkl'
SCALER_PATH  = 'models/scaler.pkl'
FEAT_PATH    = 'models/feature_cols.pkl'
SDAHU_DIR    = 'data/sdahu/'
RANDOM_STATE = 42

# ── OFFICIAL MAPPING from PDF Table 2 ─────────────────────────────────────────
COL_MAP = {
    # Supply fan
    'SF_CS'     : 'AHU: Supply Air Fan Speed Control Signal',
    'SF_SPD'    : 'AHU: Supply Air Fan Speed Position',
    'SF_SPD_DM' : 'AHU: Supply Air Fan Speed Demand Mode',
    'SF_WAT'    : 'AHU: Supply Air Fan Power',
    # Return fan
    'RF_CS'     : 'AHU: Return Air Fan Speed Control Signal',
    'RF_SPD'    : 'AHU: Return Air Fan Speed Position',
    'RF_SPD_DM' : 'AHU: Return Air Fan Speed Demand Mode',
    'RF_WAT'    : 'AHU: Return Air Fan Power',
    # Supply air
    'SA_TEMP'   : 'AHU: Supply Air Temperature',
    'SA_TEMPSPT': 'AHU: Supply Air Temperature Setpoint',
    'SA_CFM'    : 'AHU: Supply Air Flow Rate',
    'SA_SP'     : 'AHU: Supply Air Duct Static Pressure',
    'SA_SPSPT'  : 'AHU: Supply Air Duct Static Pressure Setpoint',
    # Return air
    'RA_TEMP'   : 'AHU: Return Air Temperature',
    'RA_CFM'    : 'AHU: Return Air Flow Rate',
    'RA_DMPR'   : 'AHU: Return Air Damper Position',
    'RA_DMPR_DM': 'AHU: Return Air Damper Demand Mode',
    # Outdoor air
    'OA_TEMP'   : 'AHU: Outdoor Air Temperature',
    'OA_CFM'    : 'AHU: Outdoor Air Flow Rate',
    'OA_DMPR'   : 'AHU: Outdoor Air Damper Position',
    'OA_DMPR_DM': 'AHU: Outdoor Air Damper Demand Mode',
    # Mixed air
    'MA_TEMP'   : 'AHU: Mixed Air Temperature',
    # Cooling coil valve
    'CHWC_VLV'  : 'AHU: Cooling Coil Valve Position',
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Demand Mode',
    # System
    'SYS_CTL'   : 'AHU: System Control Mode',
    # Zone temperatures
    'ZONE_TEMP_1': 'AHU: Zone Temperature 1',
    'ZONE_TEMP_2': 'AHU: Zone Temperature 2',
    'ZONE_TEMP_3': 'AHU: Zone Temperature 3',
    'ZONE_TEMP_4': 'AHU: Zone Temperature 4',
    'ZONE_TEMP_5': 'AHU: Zone Temperature 5',
}

FAULT_FILES = {
    0: ['AHU_annual.csv'],
    1: ['oa_bias_2_annual.csv', 'oa_bias_-2_annual.csv',
        'oa_bias_4_annual.csv', 'oa_bias_-4_annual.csv'],
    2: ['coi_leakage_025_annual.csv', 'coi_leakage_040_annual.csv'],
    3: ['coi_stuck_025_annual.csv',   'coi_stuck_050_annual.csv'],
    4: ['damper_stuck_025_annual.csv','damper_stuck_075_annual.csv'],
}

WINDOW_SIZES = [10, 30, 60]

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer(df):
    raw = [c for c in df.columns if c != 'label']
    frames = [df[raw].copy()]
    for w in WINDOW_SIZES:
        rm = df[raw].rolling(w, min_periods=1).mean()
        rs = df[raw].rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    feat = pd.concat(frames, axis=1)

    # Derived features — same as thesis
    sa = 'AHU: Supply Air Temperature'
    ra = 'AHU: Return Air Temperature'
    oa = 'AHU: Outdoor Air Temperature'
    ma = 'AHU: Mixed Air Temperature'
    vl = 'AHU: Cooling Coil Valve Position'
    fs = 'AHU: Supply Air Fan Speed Control Signal'
    od = 'AHU: Outdoor Air Damper Position'

    if sa in df.columns and ra in df.columns:
        feat['Temp_supply_return_diff'] = df[sa] - df[ra]
    if oa in df.columns and sa in df.columns:
        feat['Temp_oa_supply_diff']     = df[oa] - df[sa]
    if ma in df.columns and ra in df.columns:
        feat['Temp_mixed_return_diff']  = df[ma] - df[ra]
    if vl in df.columns:
        feat['Valve_differential']      = df[vl]
    if fs in df.columns and od in df.columns:
        denom = df[fs].replace(0, np.nan)
        feat['Fan_damper_ratio']        = (df[od] / denom).fillna(0)

    return feat.fillna(0)

def compute_psi(ref, tgt, n_bins=10, eps=1e-6):
    lo = min(ref.min(), tgt.min())
    hi = max(ref.max(), tgt.max())
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    rp = np.histogram(ref, bins=bins)[0].astype(float) + eps
    tp = np.histogram(tgt, bins=bins)[0].astype(float) + eps
    rp /= rp.sum(); tp /= tp.sum()
    return float(np.sum((tp - rp) * np.log(tp / rp)))

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_all(feat_cols):
    frames = []
    for label, files in FAULT_FILES.items():
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
            lbl = df.pop('label')
            feat = engineer(df)
            feat['label'] = lbl.values
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows · class {label}")
    return pd.concat(frames, ignore_index=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU CROSS-BUILDING EXPERIMENT v3")
    print("Official column mapping from PDF Table 2")
    print("=" * 65)

    print("\nStep 1 — Loading thesis artefacts...")
    model     = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(FEAT_PATH)
    print(f"  Model: {model.n_estimators} trees · {model.n_features_in_} features")

    print("\nStep 2 — Loading SD-AHU data (~2 min)...")
    data = load_all(feat_cols)
    y    = data['label'].values
    feat = data.drop(columns=['label'])

    print(f"\nStep 3 — Feature alignment...")
    common  = [c for c in feat_cols if c in feat.columns]
    missing = [c for c in feat_cols if c not in feat.columns]
    print(f"  Thesis expects:  {len(feat_cols)}")
    print(f"  Matched:         {len(common)}")
    print(f"  Zero-filled:     {len(missing)}")
    if missing:
        print(f"\n  Still zero-filled (thesis uses these, SD-AHU lacks them):")
        for m in missing:
            print(f"    - {m}")

    X = pd.DataFrame(0.0, index=feat.index, columns=feat_cols)
    for c in common:
        X[c] = feat[c].values

    print("\nStep 4 — Scaling with frozen thesis scaler...")
    X_scaled = scaler.transform(X)

    print("\nStep 5 — PSI analysis...")
    rng = np.random.default_rng(RANDOM_STATE)
    psi_scores = {}
    for i, col in enumerate(feat_cols):
        ref = rng.normal(scaler.mean_[i], max(scaler.scale_[i], 1e-6), 5000)
        tgt = X.iloc[:, i].values
        psi_scores[col] = compute_psi(ref, tgt)

    psi_s = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 PSI features (SD-AHU vs thesis training distribution):")
    for fn, psi in psi_s.head(10).items():
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn[:50]:<50} PSI={psi:.3f}  {tag}")

    print("\n  Thesis key features (for comparison):")
    key_kws = {
        'Supply Air Fan Speed Control Signal' : 'Fan speed      (thesis=2.63)',
        'Outdoor Air Temperature'             : 'OA temperature (thesis=2.45)',
        'Return Air Damper'                   : 'Return air damper',
    }
    for kw, label in key_kws.items():
        hits = [(k, v) for k, v in psi_scores.items()
                if kw in k and '_rm' not in k and '_rs' not in k]
        for k, v in hits[:1]:
            tag = "EXTREME" if v > 0.5 else "HIGH" if v > 0.2 else "LOW"
            print(f"    {label:<35} PSI={v:.3f}  {tag}")

    print("\nStep 6 — Cross-building evaluation (3-class)...")
    np.random.seed(RANDOM_STATE)
    mask3 = np.isin(y, [0, 1, 2])
    X3, y3 = X_scaled[mask3], y[mask3]
    idx3 = np.random.choice(len(y3), min(len(y3), 15000), replace=False)
    yp3  = model.predict(X3[idx3])
    f1_3 = f1_score(y3[idx3], yp3, average='macro', zero_division=0)

    print(f"\n  3-class macro-F1: {f1_3:.4f}")
    print(classification_report(y3[idx3], yp3,
          target_names=['Healthy','Fault-A (OA bias)','Fault-B (valve)'],
          zero_division=0))

    print("Step 6b — 4-class evaluation...")
    yp_all = model.predict(X_scaled)
    y4 = np.where(y  >= 3, 3, y)
    p4 = np.where(yp_all >= 3, 3, yp_all)
    f1_4 = f1_score(y4, p4, average='macro', zero_division=0)
    print(f"  4-class macro-F1: {f1_4:.4f}")

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Thesis in-sample F1:                    0.9923")
    print(f"  Thesis CB F1 (SZVAV + MZVAV-2-2):       0.331")
    print(f"  SD-AHU CB F1 3-class (Healthy/A/B):     {f1_3:.3f}")
    print(f"  SD-AHU CB F1 4-class (all faults):      {f1_4:.3f}")
    print(f"  Features matched:                        {len(common)}/{len(feat_cols)}")
    print()

    gap = f1_3 - 0.331
    if   gap >  0.05:
        verdict = "BETTER — SD-AHU control logic closer to training buildings"
        interp  = ("The generalisation gap is smaller on this building. "
                   "This suggests the ~0.33 result in the thesis may represent "
                   "a building with particularly extreme shift.")
    elif gap < -0.05:
        verdict = "WORSE — SD-AHU has more extreme distribution shift"
        interp  = ("The gap is even larger on this building. Combined with the "
                   "thesis result, this shows the 0.33 thesis result was not "
                   "the worst case — generalisation failure is widespread.")
    else:
        verdict = "CONSISTENT — ~0.33 gap reproduced on a third building"
        interp  = ("The cross-building generalisation gap is reproducible. "
                   "This directly strengthens Contribution C1 — the gap is not "
                   "specific to one building pair but a systematic pattern.")

    print(f"  Verdict: {verdict}")
    print(f"  Interpretation: {interp}")

    results = {
        'experiment'       : 'sdahu_cross_building_v3',
        'f1_3class'        : round(float(f1_3), 4),
        'f1_4class'        : round(float(f1_4), 4),
        'thesis_cb_f1'     : 0.331,
        'gap_vs_thesis'    : round(float(gap),  4),
        'features_matched' : len(common),
        'features_missing' : len(missing),
        'missing_features' : missing,
        'top5_psi'         : psi_s.head(5).round(3).to_dict(),
        'verdict'          : verdict,
    }
    os.makedirs('results', exist_ok=True)
    out = 'results/sdahu_experiment_v3.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")
    print("=" * 65)

if __name__ == '__main__':
    main()
