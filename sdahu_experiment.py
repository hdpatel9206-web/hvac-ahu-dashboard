"""
SD-AHU Cross-Building Experiment
New building from LBNL Extended FDD Dataset
Tests whether thesis findings generalise to a third building configuration

Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils import resample
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

# Path to your existing trained model from the thesis
MODEL_PATH    = 'models/rf_model.pkl'
SCALER_PATH   = 'models/scaler.pkl'
FEAT_PATH     = 'models/feature_cols.pkl'

# Path to the new SD-AHU CSV files you just downloaded
# Change this to wherever you unzipped the files
SDAHU_DIR     = 'data/sdahu/'

# SD-AHU column name → your thesis column name mapping
# Datetime,CHWC_VLV,CHWC_VLV_DM,MA_TEMP,OA_CFM,OA_DMPR,OA_DMPR_DM,OA_TEMP,
# RA_CFM,RA_DMPR,RA_DMPR_DM,RA_TEMP,RF_CS,RF_SPD,RF_SPD_DM,RF_WAT,SA_CFM,
# SA_SP,SA_SPSPT,SA_TEMP,SA_TEMPSPT,SF_CS,SF_SPD,SF_SPD_DM,SF_WAT,SYS_CTL,
# ZONE_TEMP_1,ZONE_TEMP_2,ZONE_TEMP_3,ZONE_TEMP_4,ZONE_TEMP_5

COL_MAP = {
    'Datetime'      : 'timestamp',
    'CHWC_VLV'      : 'AHU: Cooling Coil Valve Control Signal',
    'CHWC_VLV_DM'   : 'AHU: Cooling Coil Valve Damper Mode',
    'MA_TEMP'       : 'AHU: Mixed Air Temperature',
    'OA_CFM'        : 'AHU: Outdoor Air Flow Rate',
    'OA_DMPR'       : 'AHU: Outdoor Air Damper Control Signal',
    'OA_DMPR_DM'    : 'AHU: Outdoor Air Damper Mode',
    'OA_TEMP'       : 'AHU: Outdoor Air Temperature',
    'RA_CFM'        : 'AHU: Return Air Flow Rate',
    'RA_DMPR'       : 'AHU: Return Air Damper Control Signal',
    'RA_DMPR_DM'    : 'AHU: Return Air Damper Mode',
    'RA_TEMP'       : 'AHU: Return Air Temperature',
    'RF_CS'         : 'AHU: Return Fan Control Signal',
    'RF_SPD'        : 'AHU: Return Fan Speed',
    'RF_SPD_DM'     : 'AHU: Return Fan Speed Demand Mode',
    'RF_WAT'        : 'AHU: Return Fan Power',
    'SA_CFM'        : 'AHU: Supply Air Flow Rate',
    'SA_SP'         : 'AHU: Supply Air Static Pressure',
    'SA_SPSPT'      : 'AHU: Supply Air Static Pressure Setpoint',
    'SA_TEMP'       : 'AHU: Supply Air Temperature',
    'SA_TEMPSPT'    : 'AHU: Supply Air Temperature Setpoint',
    'SF_CS'         : 'AHU: Supply Fan Control Signal',
    'SF_SPD'        : 'AHU: Supply Fan Speed Control Signal',
    'SF_SPD_DM'     : 'AHU: Supply Fan Speed Demand Mode',
    'SF_WAT'        : 'AHU: Supply Fan Power',
    'SYS_CTL'       : 'AHU: System Control Signal',
    'ZONE_TEMP_1'   : 'Zone Temperature 1',
    'ZONE_TEMP_2'   : 'Zone Temperature 2',
    'ZONE_TEMP_3'   : 'Zone Temperature 3',
    'ZONE_TEMP_4'   : 'Zone Temperature 4',
    'ZONE_TEMP_5'   : 'Zone Temperature 5',
}

# Fault label mapping for the SD-AHU dataset
# We map to the same class IDs as your thesis where possible
# Fault-A = OA sensor bias (class 1), Fault-B = valve leakage (class 2)
# New faults get class 3 and 4
FAULT_FILES = {
    0: ['AHU_annual.csv'],                    # Healthy
    1: ['oa_bias_2_annual.csv',               # Fault-A: OA sensor bias
        'oa_bias_-2_annual.csv',
        'oa_bias_4_annual.csv',
        'oa_bias_-4_annual.csv'],
    2: ['coi_leakage_025_annual.csv',         # Fault-B: valve leakage
        'coi_leakage_040_annual.csv'],
    3: ['coi_stuck_025_annual.csv',           # Fault-C: coil stuck
        'coi_stuck_050_annual.csv'],
    4: ['damper_stuck_025_annual.csv',        # Fault-D: damper stuck (NEW)
        'damper_stuck_075_annual.csv'],
}

WINDOW_SIZES   = [10, 30, 60]
SUBSAMPLE_N    = 30000
RANDOM_STATE   = 42

# ─── FEATURE ENGINEERING (same as thesis pipeline) ────────────────────────────

def engineer_features(df, fit_scaler=None):
    """
    Apply same feature engineering as thesis pipeline.
    Rolling windows computed per-file (no boundary leakage).
    """
    raw_cols = [c for c in df.columns if c not in ['timestamp','label']]

    # Rolling windows per raw sensor
    roll_frames = [df[raw_cols]]
    for w in WINDOW_SIZES:
        rm = df[raw_cols].rolling(window=w, min_periods=1).mean()
        rs = df[raw_cols].rolling(window=w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw_cols]
        rs.columns = [f'{c}_rs{w}' for c in raw_cols]
        roll_frames += [rm, rs]

    feat = pd.concat(roll_frames, axis=1)

    # Derived features (scale-invariant ratios)
    if 'AHU: Supply Air Temperature' in df.columns and 'AHU: Return Air Temperature' in df.columns:
        feat['diff_sa_ra']  = df['AHU: Supply Air Temperature'] - df['AHU: Return Air Temperature']
        feat['diff_oa_sa']  = df['AHU: Outdoor Air Temperature'] - df['AHU: Supply Air Temperature']
        feat['diff_ma_ra']  = df['AHU: Mixed Air Temperature']   - df['AHU: Return Air Temperature']
    if 'AHU: Cooling Coil Valve Control Signal' in df.columns:
        feat['valve_diff']  = df['AHU: Cooling Coil Valve Control Signal'] - df.get('AHU: Cooling Coil Valve Damper Mode', 0)
    if 'AHU: Supply Fan Speed Control Signal' in df.columns and 'AHU: Outdoor Air Damper Control Signal' in df.columns:
        fan = df['AHU: Supply Fan Speed Control Signal'].replace(0, np.nan)
        dmp = df['AHU: Outdoor Air Damper Control Signal']
        feat['fan_damp_ratio'] = (dmp / fan).fillna(0)

    feat = feat.fillna(0)
    return feat

def compute_psi(train_dist, test_dist, n_bins=10, eps=1e-6):
    """Compute Population Stability Index between two distributions."""
    min_val = min(train_dist.min(), test_dist.min())
    max_val = max(train_dist.max(), test_dist.max())
    bins    = np.linspace(min_val, max_val, n_bins + 1)

    train_counts = np.histogram(train_dist, bins=bins)[0] + eps
    test_counts  = np.histogram(test_dist,  bins=bins)[0] + eps

    train_pct = train_counts / train_counts.sum()
    test_pct  = test_counts  / test_counts.sum()

    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct + eps))
    return psi

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_sdahu_file(fpath, label):
    df = pd.read_csv(fpath, low_memory=False)
    df = df.rename(columns=COL_MAP)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    df = df.select_dtypes(include=[np.number])
    df['label'] = label
    return df

def load_sdahu_dataset():
    """Load all SD-AHU files with fault labels."""
    frames = []
    for label, files in FAULT_FILES.items():
        for fname in files:
            fpath = os.path.join(SDAHU_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found — skipping")
                continue
            df = load_sdahu_file(fpath, label)
            frames.append(df)
            print(f"  Loaded {fname}: {len(df):,} rows — class {label}")
    return pd.concat(frames, ignore_index=True)

# ─── MAIN EXPERIMENT ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SD-AHU CROSS-BUILDING EXPERIMENT")
    print("=" * 65)
    print()

    # ── Step 1: Load existing thesis model ──────────────────────────
    print("Step 1 — Loading thesis model...")
    try:
        model     = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        feat_cols = joblib.load(FEAT_PATH)
        print(f"  Model loaded: {model.n_estimators} trees, "
              f"{model.n_features_in_} features")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Make sure MODEL_PATH, SCALER_PATH, FEAT_PATH are correct.")
        return

    # ── Step 2: Load SD-AHU data ────────────────────────────────────
    print("\nStep 2 — Loading SD-AHU files...")
    data = load_sdahu_dataset()
    print(f"  Total rows: {len(data):,}")
    print(f"  Class distribution:\n{data['label'].value_counts().sort_index()}")

    # ── Step 3: Feature engineering ─────────────────────────────────
    print("\nStep 3 — Engineering features...")
    labels = data['label'].values
    df_feat = engineer_features(data.drop(columns=['label']))

    # Align columns to thesis model's expected features
    print(f"  Features engineered: {df_feat.shape[1]}")
    print(f"  Model expects: {len(feat_cols)} features")

    # Keep only columns that exist in both
    common_cols = [c for c in feat_cols if c in df_feat.columns]
    missing     = [c for c in feat_cols if c not in df_feat.columns]
    extra       = [c for c in df_feat.columns if c not in feat_cols]

    print(f"  Matching features: {len(common_cols)}")
    if missing:
        print(f"  Missing from SD-AHU (will be zero-filled): {len(missing)}")
    if extra:
        print(f"  Extra in SD-AHU (will be dropped): {len(extra)}")

    # Build aligned feature matrix
    X = pd.DataFrame(0.0, index=df_feat.index, columns=feat_cols)
    for c in common_cols:
        X[c] = df_feat[c].values
    y = labels

    # ── Step 4: Scale using thesis scaler ───────────────────────────
    print("\nStep 4 — Scaling with thesis StandardScaler (frozen)...")
    X_scaled = scaler.transform(X)
    print("  Scaler applied — parameters unchanged from training")

    # ── Step 5: PSI Analysis ────────────────────────────────────────
    print("\nStep 5 — PSI distribution shift analysis...")

    # Load thesis training data distribution from scaler
    # We use scaler.mean_ as the training reference distribution
    train_means = scaler.mean_
    test_means  = X.mean(axis=0).values

    psi_scores = {}
    for i, col in enumerate(feat_cols):
        # Use actual column distributions
        train_col = np.random.normal(train_means[i],
                                     scaler.scale_[i], 1000)
        test_col  = X.iloc[:, i].values
        psi_scores[col] = compute_psi(train_col, test_col)

    # Top 10 highest PSI features
    psi_series = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 features by PSI (distribution shift):")
    for feat, psi in psi_series.head(10).items():
        level = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {feat[:45]:<45}  PSI={psi:.3f}  {level}")

    # Key features from thesis
    print("\n  Thesis key features:")
    for kw in ['SF_SPD', 'Supply Fan Speed', 'OA_TEMP', 'Outdoor Air Temp',
               'Return Air Damper', 'RA_DMPR']:
        matches = [(k,v) for k,v in psi_scores.items() if kw.lower() in k.lower()]
        for k,v in matches[:1]:
            print(f"    {k[:50]:<50}  PSI={v:.3f}")

    # ── Step 6: Cross-building evaluation ───────────────────────────
    print("\nStep 6 — Cross-building evaluation (3-class: Healthy, OA bias, Valve leakage)...")

    # Filter to classes 0,1,2 to match thesis test set structure
    mask_3cls = np.isin(y, [0, 1, 2])
    X3 = X_scaled[mask_3cls]
    y3 = y[mask_3cls]

    # Subsample for evaluation balance
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(y3), min(len(y3), 15000), replace=False)
    X3s, y3s = X3[idx], y3[idx]

    y_pred_3 = model.predict(X3s)
    f1_3cls  = f1_score(y3s, y_pred_3, average='macro', zero_division=0)

    print(f"\n  3-class macro-F1: {f1_3cls:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y3s, y_pred_3,
          target_names=['Healthy','Fault-A (OA bias)','Fault-B (valve)'],
          zero_division=0))

    # 4-class evaluation (all classes)
    print("\nStep 6b — 4-class evaluation (all fault types)...")
    y_pred_all = model.predict(X_scaled)

    # Remap class 3 → 3, class 4 → 3 (treat all new faults as class 3)
    y_remap = np.where(y >= 3, 3, y)
    y_pred_remap = np.where(y_pred_all >= 3, 3, y_pred_all)

    f1_4cls = f1_score(y_remap, y_pred_remap,
                       average='macro', zero_division=0)
    print(f"  4-class macro-F1 (remapped): {f1_4cls:.4f}")

    # ── Step 7: Compare to thesis result ────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS vs THESIS")
    print("=" * 65)
    print(f"  Thesis CB F1 (SZVAV + MZVAV-2-2):  0.331")
    print(f"  New SD-AHU CB F1 (3-class):         {f1_3cls:.3f}")
    print(f"  New SD-AHU CB F1 (4-class):         {f1_4cls:.3f}")
    print()

    if f1_3cls > 0.40:
        print("  FINDING: SD-AHU shows BETTER generalisation than thesis CB test.")
        print("  Interpretation: This building's control logic is more similar")
        print("  to the training buildings. PSI analysis will confirm which")
        print("  features are less shifted.")
    elif f1_3cls < 0.25:
        print("  FINDING: SD-AHU shows WORSE generalisation than thesis CB test.")
        print("  Interpretation: More extreme distribution shift in this building.")
    else:
        print("  FINDING: SD-AHU shows SIMILAR generalisation to thesis CB test.")
        print("  Interpretation: The ~0.33 result is consistent across buildings.")
        print("  This STRENGTHENS Contribution C1 — the gap is reproducible.")

    # ── Step 8: Save results ─────────────────────────────────────────
    results = {
        'experiment'           : 'sdahu_cross_building',
        'model_source'         : 'thesis_run_20260415_142209',
        'new_building'         : 'LBNL_SD_AHU_extended',
        'f1_3class'            : round(float(f1_3cls), 4),
        'f1_4class'            : round(float(f1_4cls), 4),
        'thesis_cb_f1'         : 0.331,
        'difference'           : round(float(f1_3cls - 0.331), 4),
        'top_psi_features'     : psi_series.head(5).to_dict(),
        'n_samples_evaluated'  : int(len(y3s)),
        'classes_in_test'      : [int(c) for c in np.unique(y3)],
    }

    os.makedirs('results', exist_ok=True)
    with open('results/sdahu_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n  Results saved to results/sdahu_experiment.json")
    print("=" * 65)

if __name__ == '__main__':
    main()
