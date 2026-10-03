import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

BASE   = r'C:\Users\hrslp\Desktop\thesis'
MODEL  = os.path.join(BASE, 'models', 'rf_model.pkl')
SCALER = os.path.join(BASE, 'models', 'scaler.pkl')
FEATS  = os.path.join(BASE, 'models', 'feature_cols.pkl')
DATA   = os.path.join(BASE, 'data')

print("Loading saved model and cross-building data...")
rf        = joblib.load(MODEL)
scaler    = joblib.load(SCALER)
feat_cols = joblib.load(FEATS)

LABEL_MAP = {0:'Healthy',1:'Fault-A',2:'Fault-B',3:'Fault-C'}
FEATS_RAW = [
    'AHU: Supply Air Temperature','AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature','AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status','AHU: Supply Air Fan Speed Control Signal',
    'AHU: Cooling Coil Valve Control Signal','AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

def load_eng(fname, fc):
    df = pd.read_csv(os.path.join(DATA, fname), parse_dates=['Datetime'])
    df = df.rename(columns={'Datetime':'Timestamp'})
    df.columns = [c.strip() for c in df.columns]
    for c in FEATS_RAW:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.ffill().fillna(0)
    df['label'] = df['Fault Detection Ground Truth'].apply(lambda x: fc if x==1 else 0)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    for col in FEATS_RAW:
        if col in df.columns:
            for w in [10,30,60]:
                df[f'{col}_rm{w}'] = df[col].rolling(w,min_periods=1).mean()
                df[f'{col}_rs{w}'] = df[col].rolling(w,min_periods=1).std().fillna(0)
    df['Temp_supply_return_diff'] = df['AHU: Supply Air Temperature'] - df['AHU: Return Air Temperature']
    df['Temp_outdoor_supply_diff'] = df['AHU: Supply Air Temperature'] - df['AHU: Outdoor Air Temperature']
    df['Temp_mixed_return_diff'] = df['AHU: Mixed Air Temperature'] - df['AHU: Return Air Temperature']
    df['Valve_diff'] = df['AHU: Cooling Coil Valve Control Signal'] - df['AHU: Heating Coil Valve Control Signal']
    df['Fan_damper_ratio'] = df['AHU: Supply Air Fan Speed Control Signal'] - df['AHU: Outdoor Air Damper Control Signal']
    return df

print("Loading cross-building data...")
cb = pd.concat([
    load_eng('MZVAV-2-2.csv', 2),
    load_eng('SZVAV.csv', 3)
]).reset_index(drop=True)

X_cb = cb[feat_cols].values
y_cb = cb['label'].values

# Baseline — no adaptation
X_cb_sc = scaler.transform(X_cb)
y_pred_base = rf.predict(X_cb_sc)
f1_base = f1_score(y_cb, y_pred_base, average='macro', zero_division=0)
print(f"\nBaseline cross-building F1 (no adaptation): {f1_base:.4f}")

# Load original training data to combine with adaptation data
print("\nLoading original training data for combined fine-tuning...")
from sklearn.model_selection import train_test_split

orig = pd.concat([
    load_eng('MZVAV-1.csv',   1),
    load_eng('MZVAV-2-1.csv', 2),
    load_eng('SZCAV.csv',     3)
]).reset_index(drop=True)

# Subsample original to 30k per class
orig_parts = []
for lbl in orig['label'].unique():
    subset = orig[orig['label'] == lbl]
    n = min(len(subset), 30000)
    orig_parts.append(subset.sample(n=n, random_state=42))
orig_sub = pd.concat(orig_parts)

X_orig = scaler.transform(orig_sub[feat_cols].values)
y_orig = orig_sub['label'].values

print(f"Original training size: {len(X_orig):,}")

# Fine-tuning experiment — combine original + adaptation data
print("\nFine-tuning experiment (original + adaptation data):")
print(f"{'Adapt %':<12} {'Adapt N':<12} {'F1 Adapted':<14} {'Improvement'}")
print("-" * 55)

results = []
for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
    n_adapt = int(len(X_cb) * pct)

    X_adapt = X_cb_sc[:n_adapt]
    y_adapt = y_cb[:n_adapt]

    X_test  = X_cb_sc[n_adapt:]
    y_test  = y_cb[n_adapt:]

    if len(np.unique(y_test)) < 2:
        continue

    # Combine original training + adaptation data
    X_combined = np.vstack([X_orig, X_adapt])
    y_combined = np.concatenate([y_orig, y_adapt])

    rf_adapted = RandomForestClassifier(
        n_estimators=200, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf_adapted.fit(X_combined, y_combined)

    y_pred_adapted = rf_adapted.predict(X_test)
    f1_adapted = f1_score(y_test, y_pred_adapted,
                          average='macro', zero_division=0,
                          labels=sorted(np.unique(np.concatenate([y_test, y_pred_adapted]))))

    improvement = f1_adapted - f1_base
    print(f"  {pct*100:.0f}%         {n_adapt:<12} {f1_adapted:<14.4f} {improvement:+.4f}")
    results.append({'pct': pct, 'n_adapt': n_adapt,
                    'f1_adapted': f1_adapted, 'improvement': improvement})

print(f"\nBaseline (0% adaptation): {f1_base:.4f}")
best = max(results, key=lambda x: x['f1_adapted'])
print(f"Best result: {best['pct']*100:.0f}% adaptation → F1 = {best['f1_adapted']:.4f} ({best['improvement']:+.4f})")
print("\nDone.")