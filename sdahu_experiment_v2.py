"""
SD-AHU Cross-Building Experiment v2
Fixed column mapping — all 70 features properly aligned

Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
warnings.filterwarnings('ignore')

# ─── PATHS ───────────────────────────────────────────────────────────────────
MODEL_PATH  = 'models/rf_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEAT_PATH   = 'models/feature_cols.pkl'
SDAHU_DIR   = 'data/sdahu/'
RANDOM_STATE = 42

# ─── FULL COLUMN MAPPING ─────────────────────────────────────────────────────
# SD-AHU short code → thesis long name
# Every sensor that exists in SD-AHU is mapped here
COL_MAP = {
    # Cooling coil valve
    'CHWC_VLV'      : 'AHU: Cooling Coil Valve Control Signal',
    'CHWC_VLV_DM'   : 'AHU: Cooling Coil Valve Command',
    # Mixed air
    'MA_TEMP'       : 'AHU: Mixed Air Temperature',
    # Outdoor air
    'OA_CFM'        : 'AHU: Outdoor Air Flow Rate',
    'OA_DMPR'       : 'AHU: Outdoor Air Damper Control Signal',
    'OA_DMPR_DM'    : 'AHU: Outdoor Air Damper Command',
    'OA_TEMP'       : 'AHU: Outdoor Air Temperature',
    # Return air
    'RA_CFM'        : 'AHU: Return Air Flow Rate',
    'RA_DMPR'       : 'AHU: Return Air Damper Control Signal',
    'RA_DMPR_DM'    : 'AHU: Return Air Damper Command',
    'RA_TEMP'       : 'AHU: Return Air Temperature',
    # Return fan
    'RF_CS'         : 'AHU: Return Air Fan Status',
    'RF_SPD'        : 'AHU: Return Air Fan Speed Control Signal',
    'RF_SPD_DM'     : 'AHU: Return Fan Speed Command',
    'RF_WAT'        : 'AHU: Return Fan Power',
    # Supply air
    'SA_CFM'        : 'AHU: Supply Air Flow Rate',
    'SA_SP'         : 'AHU: Supply Air Duct Static Pressure',
    'SA_SPSPT'      : 'AHU: Supply Air Duct Static Pressure Setpoint',
    'SA_TEMP'       : 'AHU: Supply Air Temperature',
    'SA_TEMPSPT'    : 'AHU: Supply Air Temperature Setpoint',
    # Supply fan
    'SF_CS'         : 'AHU: Supply Air Fan Status',
    'SF_SPD'        : 'AHU: Supply Air Fan Speed Control Signal',
    'SF_SPD_DM'     : 'AHU: Supply Fan Speed Command',
    'SF_WAT'        : 'AHU: Supply Fan Power',
    # System
    'SYS_CTL'       : 'AHU: System Mode',
    # Zone temps — map to occupancy proxy
    'ZONE_TEMP_1'   : 'Zone Temperature 1',
    'ZONE_TEMP_2'   : 'Zone Temperature 2',
    'ZONE_TEMP_3'   : 'Zone Temperature 3',
    'ZONE_TEMP_4'   : 'Zone Temperature 4',
    'ZONE_TEMP_5'   : 'Zone Temperature 5',
}

# Fault files with class labels
FAULT_FILES = {
    0: ['AHU_annual.csv'],
    1: ['oa_bias_2_annual.csv', 'oa_bias_-2_annual.csv',
        'oa_bias_4_annual.csv', 'oa_bias_-4_annual.csv'],
    2: ['coi_leakage_025_annual.csv', 'coi_leakage_040_annual.csv'],
    3: ['coi_stuck_025_annual.csv',   'coi_stuck_050_annual.csv'],
    4: ['damper_stuck_025_annual.csv','damper_stuck_075_annual.csv'],
}

WINDOW_SIZES = [10, 30, 60]

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def engineer_features_per_file(df):
    """Same pipeline as thesis — per-file rolling, no boundary leakage."""
    raw_cols = [c for c in df.columns if c != 'label']
    frames = [df[raw_cols].copy()]

    for w in WINDOW_SIZES:
        rm = df[raw_cols].rolling(window=w, min_periods=1).mean()
        rs = df[raw_cols].rolling(window=w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw_cols]
        rs.columns = [f'{c}_rs{w}' for c in raw_cols]
        frames += [rm, rs]

    feat = pd.concat(frames, axis=1)

    # Derived features
    sa = 'AHU: Supply Air Temperature'
    ra = 'AHU: Return Air Temperature'
    oa = 'AHU: Outdoor Air Temperature'
    ma = 'AHU: Mixed Air Temperature'
    vl = 'AHU: Cooling Coil Valve Control Signal'
    fs = 'AHU: Supply Air Fan Speed Control Signal'
    od = 'AHU: Outdoor Air Damper Control Signal'

    if sa in df.columns and ra in df.columns:
        feat['Temp_supply_return_diff'] = df[sa] - df[ra]
    if oa in df.columns and sa in df.columns:
        feat['Temp_oa_supply_diff'] = df[oa] - df[sa]
    if ma in df.columns and ra in df.columns:
        feat['Temp_mixed_return_diff'] = df[ma] - df[ra]
    if vl in df.columns:
        feat['Valve_differential'] = df[vl]
    if fs in df.columns and od in df.columns:
        denom = df[fs].replace(0, np.nan)
        feat['Fan_damper_ratio'] = (df[od] / denom).fillna(0)

    return feat.fillna(0)

def compute_psi(ref, tgt, n_bins=10, eps=1e-6):
    lo = min(ref.min(), tgt.min())
    hi = max(ref.max(), tgt.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    rp = np.histogram(ref, bins=bins)[0].astype(float) + eps
    tp = np.histogram(tgt, bins=bins)[0].astype(float) + eps
    rp /= rp.sum(); tp /= tp.sum()
    return float(np.sum((tp - rp) * np.log(tp / rp)))

# ─── LOAD SD-AHU ─────────────────────────────────────────────────────────────

def load_all():
    frames = []
    for label, files in FAULT_FILES.items():
        for fname in files:
            fpath = os.path.join(SDAHU_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found"); continue
            df = pd.read_csv(fpath, low_memory=False)
            # Rename columns
            df = df.rename(columns=COL_MAP)
            # Drop timestamp if present
            df = df.drop(columns=[c for c in df.columns
                                   if 'datetime' in c.lower() or
                                      'timestamp' in c.lower()], errors='ignore')
            df = df.select_dtypes(include=[np.number])
            df['label'] = label
            # Engineer features PER FILE (prevents boundary leakage)
            lbl = df.pop('label')
            feat = engineer_features_per_file(df)
            feat['label'] = lbl.values
            frames.append(feat)
            print(f"  {fname}: {len(feat):,} rows, class {label}")
    return pd.concat(frames, ignore_index=True)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU CROSS-BUILDING EXPERIMENT v2 — Fixed Column Mapping")
    print("=" * 65)

    # Load thesis artefacts
    print("\nStep 1 — Loading thesis model...")
    model     = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(FEAT_PATH)
    print(f"  {model.n_estimators} trees · {model.n_features_in_} features")

    # Load and engineer SD-AHU data
    print("\nStep 2 — Loading SD-AHU data (this takes ~2 min)...")
    data  = load_all()
    y     = data['label'].values
    feat  = data.drop(columns=['label'])

    # Align to thesis feature columns
    print(f"\nStep 3 — Aligning {feat.shape[1]} engineered features "
          f"to {len(feat_cols)} thesis features...")
    common  = [c for c in feat_cols if c in feat.columns]
    missing = [c for c in feat_cols if c not in feat.columns]
    print(f"  Matched: {len(common)}  |  Zero-filled: {len(missing)}")
    if missing:
        print("  Zero-filled features:")
        for m in missing:
            print(f"    - {m}")

    X = pd.DataFrame(0.0, index=feat.index, columns=feat_cols)
    for c in common:
        X[c] = feat[c].values

    # Scale
    print("\nStep 4 — Applying frozen thesis StandardScaler...")
    X_scaled = scaler.transform(X)

    # PSI analysis using scaler reference distribution
    print("\nStep 5 — PSI analysis...")
    psi_scores = {}
    for i, col in enumerate(feat_cols):
        ref = np.random.default_rng(42).normal(
              scaler.mean_[i], scaler.scale_[i], 5000)
        tgt = X.iloc[:, i].values
        psi_scores[col] = compute_psi(ref, tgt)

    psi_s = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 PSI features:")
    for feat_name, psi in psi_s.head(10).items():
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {feat_name[:48]:<48} PSI={psi:.3f}  {tag}")

    thesis_keys = {
        'Supply Air Fan Speed Control Signal' : 'Fan speed (thesis PSI=2.63)',
        'Outdoor Air Temperature'             : 'OA temperature (thesis PSI=2.45)',
        'Return Air Damper Control Signal'    : 'Return air damper',
    }
    print("\n  Thesis key features in SD-AHU building:")
    for kw, label in thesis_keys.items():
        matches = [(k, v) for k, v in psi_scores.items() if kw in k and '_rm' not in k and '_rs' not in k]
        for k, v in matches[:1]:
            print(f"    {label:<40} PSI={v:.3f}")

    # Evaluation — 3 classes (Healthy, Fault-A, Fault-B)
    print("\nStep 6 — Cross-building evaluation...")
    np.random.seed(RANDOM_STATE)

    mask3 = np.isin(y, [0, 1, 2])
    X3, y3 = X_scaled[mask3], y[mask3]
    idx3 = np.random.choice(len(y3), min(len(y3), 15000), replace=False)
    y_pred3 = model.predict(X3[idx3])
    f1_3 = f1_score(y3[idx3], y_pred3, average='macro', zero_division=0)

    print(f"\n  3-class macro-F1: {f1_3:.4f}")
    print(f"\n  Classification report (3-class):")
    print(classification_report(y3[idx3], y_pred3,
          target_names=['Healthy', 'Fault-A (OA bias)', 'Fault-B (valve)'],
          zero_division=0))

    # 4-class
    y_pred_all = model.predict(X_scaled)
    y4  = np.where(y  >= 3, 3, y)
    p4  = np.where(y_pred_all >= 3, 3, y_pred_all)
    f1_4 = f1_score(y4, p4, average='macro', zero_division=0)
    print(f"  4-class macro-F1: {f1_4:.4f}")

    # Summary
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  Thesis in-sample F1:              0.9923")
    print(f"  Thesis CB F1 (SZVAV+MZVAV-2-2):  0.331")
    print(f"  SD-AHU CB F1 (3-class):           {f1_3:.3f}")
    print(f"  SD-AHU CB F1 (4-class):           {f1_4:.3f}")
    print(f"  Matched features:                 {len(common)} / {len(feat_cols)}")
    print()

    gap = f1_3 - 0.331
    if   gap >  0.05: verdict = "BETTER — SD-AHU control logic closer to training buildings"
    elif gap < -0.05: verdict = "WORSE  — SD-AHU has more extreme distribution shift"
    else:             verdict = "SIMILAR — ~0.33 gap is reproducible across buildings"
    print(f"  Verdict: {verdict}")

    # Save
    results = {
        'experiment'       : 'sdahu_cross_building_v2',
        'f1_3class'        : round(float(f1_3), 4),
        'f1_4class'        : round(float(f1_4), 4),
        'thesis_cb_f1'     : 0.331,
        'gap_vs_thesis'    : round(float(f1_3 - 0.331), 4),
        'features_matched' : len(common),
        'features_missing' : len(missing),
        'missing_list'     : missing,
        'top5_psi'         : psi_s.head(5).round(3).to_dict(),
        'verdict'          : verdict,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/sdahu_experiment_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to results/sdahu_experiment_v2.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
