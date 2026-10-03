"""
CONFIDENCE COMPARISON
Run: python experiments/confidence_comparison.py
"""
import warnings, json
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RS = 42

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

# Fault labels based on filename
FILE_LABELS = {
    'MZVAV-1.csv':   1,
    'MZVAV-2-1.csv': 2,
    'SZCAV.csv':     3,
}

def engineer(df):
    avail = [c for c in RAW_COLS if c in df.columns]
    out = df[avail].copy()
    for col in avail:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in avail:
        for w in [10, 30, 60]:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    pairs = [
        ('Temp_supply_return_diff','AHU: Supply Air Temperature','AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff','AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff','AHU: Mixed Air Temperature','AHU: Return Air Temperature'),
        ('Valve_diff','AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal'),
        ('Fan_damper_ratio','AHU: Supply Air Fan Speed Control Signal','AHU: Outdoor Air Damper Control Signal'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)

def conf_stats(conf, name):
    print(f"  {name} ({len(conf):,} rows):")
    print(f"    Mean     : {conf.mean():.3f}")
    print(f"    Median   : {np.median(conf):.3f}")
    print(f"    > 0.9    : {(conf>0.9).mean()*100:.1f}%")
    print(f"    > 0.7    : {(conf>0.7).mean()*100:.1f}%")
    print(f"    < 0.6    : {(conf<0.6).mean()*100:.1f}%")
    print()

print("="*60)
print("  CONFIDENCE COMPARISON")
print("="*60)

model     = joblib.load(BASE_DIR/'models'/'rf_model.pkl')
feat_cols = joblib.load(BASE_DIR/'models'/'feature_cols.pkl')
scaler    = joblib.load(BASE_DIR/'models'/'scaler.pkl')
print(f"Model loaded — {len(feat_cols)} features expected\n")

# ── Build in-sample test set with full feature engineering ──────────────────
print("Loading and engineering training data...")
parts = []
for fname, lbl in FILE_LABELS.items():
    df = pd.read_csv(DATA_DIR / fname)
    fe = engineer(df)
    fe['label'] = lbl
    # add healthy rows — rows where all sensor values are near-zero are not
    # present here; entire file is one fault type per the ASHRAE structure
    parts.append(fe)
    print(f"  {fname}: {len(fe):,} rows → label {lbl}")

df_all = pd.concat(parts, ignore_index=True)

# Add healthy class — sample from rows across all files with low fault signal
# Since no Active Fault column, use rows from first quarter of MZVAV-1 (pre-fault)
df_base = pd.read_csv(DATA_DIR / 'MZVAV-1.csv')
fe_base = engineer(df_base)
# Use first 10% as approximate healthy baseline
n_healthy = min(30000, len(fe_base)//10)
fe_healthy = fe_base.head(n_healthy).copy()
fe_healthy['label'] = 0
df_all = pd.concat([df_all, fe_healthy], ignore_index=True)
print(f"  Healthy rows added: {n_healthy:,}")

# Subsample
subs = []
for lbl in df_all['label'].unique():
    s = df_all[df_all['label']==lbl]
    subs.append(s.sample(min(len(s), 30000), random_state=RS))
df_sub = pd.concat(subs).reset_index(drop=True)

# Use only feat_cols that exist after engineering
avail_feats = [c for c in feat_cols if c in df_sub.columns]
print(f"\nEngineered features available: {len(avail_feats)} / {len(feat_cols)}")

X = df_sub[avail_feats].fillna(0).values
y = df_sub['label'].values

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

# Pad columns to match scaler's 70 features
X_test_full = np.zeros((len(X_test), len(feat_cols)))
X_test_full[:, :len(avail_feats)] = X_test

X_test_s   = scaler.transform(X_test_full)
in_conf    = model.predict_proba(X_test_s).max(axis=1)

print(f"\n{'='*60}")
print("  RESULTS")
print(f"{'='*60}\n")

conf_stats(in_conf, "In-sample AHU (same buildings as training)")

# ── Cross-building from predictions CSV ────────────────────────────────────
df_cb     = pd.read_csv(DATA_DIR / 'ahu_predictions.csv')
prob_cols = [c for c in df_cb.columns if c.startswith('Prob_')]
cb_conf   = df_cb[prob_cols].max(axis=1).values
conf_stats(cb_conf, "Cross-building AHU (unseen buildings)")

# ── RTU out-of-distribution ─────────────────────────────────────────────────
df_rtu = pd.read_csv(DATA_DIR / 'RTU.csv').rename(columns={
    'RTU: Supply Air Temperature': 'AHU: Supply Air Temperature',
    'RTU: Return Air Temperature':  'AHU: Return Air Temperature',
    'RTU: Supply Air Fan Status':   'AHU: Supply Air Fan Status',
})
X_rtu = np.zeros((len(df_rtu), len(feat_cols)))
for i, col in enumerate(feat_cols):
    if col in df_rtu.columns:
        X_rtu[:, i] = pd.to_numeric(df_rtu[col], errors='coerce').fillna(0).values
rtu_conf  = model.predict_proba(scaler.transform(X_rtu)).max(axis=1)
rtu_preds = model.predict(scaler.transform(X_rtu))

conf_stats(rtu_conf, "RTU out-of-distribution (different equipment type)")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  {'Dataset':<42} {'Mean':>6} {'>0.7':>7} {'>0.9':>7}")
print(f"  {'-'*42} {'-'*6} {'-'*7} {'-'*7}")
for name, conf in [
    ('In-sample AHU', in_conf),
    ('Cross-building AHU', cb_conf),
    ('RTU (out-of-distribution)', rtu_conf),
]:
    print(f"  {name:<42} {conf.mean():>6.3f} "
          f"{(conf>0.7).mean()*100:>6.1f}% "
          f"{(conf>0.9).mean()*100:>6.1f}%")

print()
LABEL_MAP = {0:'Healthy',1:'Fault-A',2:'Fault-B',3:'Fault-C'}
print("RTU prediction distribution:")
for u, c in zip(*np.unique(rtu_preds, return_counts=True)):
    print(f"  {LABEL_MAP[u]}: {c:,} ({c/len(rtu_preds)*100:.1f}%)")

results = {
    'insample':      {'n':int(len(in_conf)), 'mean':round(float(in_conf.mean()),3), 'pct_07':round(float((in_conf>0.7).mean()*100),1), 'pct_09':round(float((in_conf>0.9).mean()*100),1)},
    'crossbuilding': {'n':int(len(cb_conf)), 'mean':round(float(cb_conf.mean()),3), 'pct_07':round(float((cb_conf>0.7).mean()*100),1), 'pct_09':round(float((cb_conf>0.9).mean()*100),1)},
    'rtu':           {'n':int(len(rtu_conf)),'mean':round(float(rtu_conf.mean()),3),'pct_07':round(float((rtu_conf>0.7).mean()*100),1),'pct_09':round(float((rtu_conf>0.9).mean()*100),1)},
}
(BASE_DIR/'results').mkdir(exist_ok=True)
with open(BASE_DIR/'results'/'confidence_comparison.json','w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/confidence_comparison.json")
