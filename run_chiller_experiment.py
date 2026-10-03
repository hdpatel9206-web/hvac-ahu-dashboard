"""
CHILLER EXPERIMENT RUNNER — ASHRAE RP-1043
==========================================
Loads Excel files from data/chiller/ folder structure,
engineers features, trains models, evaluates performance.

Run:
    python run_chiller_experiment.py

Output:
    results/run_YYYYMMDD_HHMMSS_CHILLER_RP1043/
"""

from __future__ import annotations
import warnings, sys
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.svm import SVC

BASE_DIR = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR = BASE_DIR / "data" / "chiller"

# ── Timestamp for this run ──
TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = BASE_DIR / "results" / f"run_{TS}_CHILLER_RP1043"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RS = 42

# ── Simplified 4-class fault map ──
FAULT_FOLDERS = {
    "Benchmark Tests":   0,
    "Condenser fouling": 1,
    "Refrigerant leak":  2,
    "Excess oil":        3,
}
LABEL_MAP = {
    0: "Healthy",
    1: "Condenser Fouling",
    2: "Refrigerant Leak",
    3: "Excess Oil",
}
SUBSAMPLE = 5000
ROLLING_WINDOWS = [10, 30, 60]

# ── Raw feature columns ──
RAW_COLS = [
    'TWE_set','TEI','TWEI','TEO','TWEO','TCI','TWCI','TCO','TWCO',
    'TSI','TSO','TBI','TBO',
    'Cond Tons','Cooling Tons','Shared Cond Tons','Cond Energy Balance',
    'Evap Tons','Shared Evap Tons','Building Tons','Evap Energy Balance',
    'kW','COP','kW/Ton','FWC','FWE','TEA','TCA',
    'TRE','PRE','TRC','PRC','TRC_sub','T_suc','Tsh_suc',
    'TR_dis','Tsh_dis','P_lift','Amps','RLA%',
    'Heat Balance (kW)','Heat Balance%','Tolerance%',
    'TO_sump','TO_feed','PO_feed','PO_net',
    'TWCD','TWED','VSS','VSL','VH','VM','VC','VE','VW',
    'TWI','TWO','THI','THO','FWW','FWH','FWB'
]

print("=" * 65)
print("  CHILLER EXPERIMENT RUNNER")
print(f"  Dataset   : CHILLER_RP1043")
print(f"  Run folder: results/run_{TS}_CHILLER_RP1043")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/5] Loading chiller data from Excel files...")

def load_folder(folder_name: str, label: int) -> pd.DataFrame:
    folder = DATA_DIR / folder_name
    if not folder.exists():
        print(f"  WARNING: folder not found — {folder}")
        return pd.DataFrame()
    
    frames = []
    for f in folder.glob("*.xls*"):
        try:
            df = pd.read_excel(f, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            # Keep only columns that exist
            available = [c for c in RAW_COLS if c in df.columns]
            if len(available) < 10:
                print(f"    WARNING: {f.name} has only {len(available)} matching columns")
                continue
            df = df[available].copy()
            df['label'] = label
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            frames.append(df)
        except Exception as e:
            print(f"    ERROR reading {f.name}: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    print(f"  {folder_name:<35}: {len(combined):>7,} rows | label={label} ({LABEL_MAP[label]})")
    return combined

all_data = []
for folder, lbl in FAULT_FOLDERS.items():
    df = load_folder(folder, lbl)
    if not df.empty:
        all_data.append(df)

if not all_data:
    print("ERROR: No data loaded. Check folder paths.")
    sys.exit(1)

df_all = pd.concat(all_data, ignore_index=True)

# Detect actual feature columns (intersection of RAW_COLS and what loaded)
actual_raw = [c for c in RAW_COLS if c in df_all.columns]
print(f"\n  Raw feature columns found: {len(actual_raw)}")
print(f"  Total rows: {len(df_all):,}")

# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[2/5] Engineering features...")

for col in actual_raw:
    for w in ROLLING_WINDOWS:
        df_all[f'{col}_rm{w}'] = df_all.groupby('label')[col].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df_all[f'{col}_rs{w}'] = df_all.groupby('label')[col].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))

# Derived features — only if both columns exist
derived = {
    'Temp_evap_diff':      ('TEO', 'TEI'),
    'Temp_cond_diff':      ('TCO', 'TCI'),
    'Pressure_lift':       ('PRC', 'PRE'),
}
for name, (a, b) in derived.items():
    if a in df_all.columns and b in df_all.columns:
        df_all[name] = df_all[a] - df_all[b]

# Final feature columns
feat_cols = [c for c in df_all.columns if c != 'label' and
             df_all[c].dtype in [np.float64, np.int64, float, int]]

print(f"  Feature columns: {len(feat_cols)}")

# ══════════════════════════════════════════════════════════════
# 3. SUBSAMPLE + SPLIT
# ══════════════════════════════════════════════════════════════
print("\n[3/5] Building train/test split...")

parts = []
for lbl in df_all['label'].unique():
    sub = df_all[df_all['label'] == lbl]
    n = min(len(sub), SUBSAMPLE)
    parts.append(sub.sample(n=n, random_state=RS))
df_sub = pd.concat(parts).reset_index(drop=True)

# Fix NaN values from rolling windows
df_sub[feat_cols] = df_sub[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
X = df_sub[feat_cols].values
y = df_sub['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  Features         : {len(feat_cols)}")

# ══════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n[4/5] Training models...")

models = {
    'Rule-based (Dummy)': DummyClassifier(strategy='stratified', random_state=RS),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RS),
    'Random Forest':      RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                                  class_weight='balanced', n_jobs=-1, random_state=RS),
}

print(f"\n  {'Model':<28} {'Macro-F1':>10} {'MCC':>10} {'Bal-Acc':>10}")
print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*10}")

results = {}
best_model = None
best_f1    = 0.0

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    ba  = balanced_accuracy_score(y_test, y_pred)
    results[name] = dict(f1=round(f1,4), mcc=round(mcc,4), bal_acc=round(ba,4))
    marker = ' <- BEST' if f1 > best_f1 else ''
    print(f"  {name:<28} {f1:>10.4f} {mcc:>10.4f} {ba:>10.4f}{marker}")
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name  = name
        y_pred_best = y_pred

# ══════════════════════════════════════════════════════════════
# 5. METRICS + SAVE
# ══════════════════════════════════════════════════════════════
print("\n[5/5] Computing metrics and saving...")

print(f"\n  === {best_name} — Classification Report ===")
print(classification_report(y_test, y_pred_best,
      target_names=list(LABEL_MAP.values()), zero_division=0))

# Bootstrap CI
f1_boot = []
rng = np.random.default_rng(RS)
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[idx])) < 2:
        continue
    f1_boot.append(f1_score(y_test[idx], y_pred_best[idx],
                             average='macro', zero_division=0))
ci_lo = round(np.percentile(f1_boot, 2.5), 4)
ci_hi = round(np.percentile(f1_boot, 97.5), 4)
print(f"  Bootstrap CI (1000): [{ci_lo}, {ci_hi}]")

# McNemar vs rule-based
from statsmodels.stats.contingency_tables import mcnemar
rf_correct    = (y_pred_best == y_test)
dummy_correct = (models['Rule-based (Dummy)'].predict(X_test_s) == y_test)
ct = np.array([[np.sum(rf_correct & dummy_correct),
                np.sum(rf_correct & ~dummy_correct)],
               [np.sum(~rf_correct & dummy_correct),
                np.sum(~rf_correct & ~dummy_correct)]])
try:
    result = mcnemar(ct, exact=False)
    print(f"  McNemar RF vs Dummy: chi2={result.statistic:.3f}, p={result.pvalue:.4f}")
except:
    print("  McNemar test: could not compute")

# Save model
model_dir = RUN_DIR / 'models'
model_dir.mkdir(exist_ok=True)
joblib.dump(best_model, model_dir / 'chiller_rf_model.pkl')
joblib.dump(scaler,     model_dir / 'chiller_scaler.pkl')
joblib.dump(feat_cols,  model_dir / 'chiller_feature_cols.pkl')

# Also copy to models/ for dashboard
models_dir = BASE_DIR / 'models'
joblib.dump(best_model, models_dir / 'chiller_rf_model.pkl')
joblib.dump(scaler,     models_dir / 'chiller_scaler.pkl')
joblib.dump(feat_cols,  models_dir / 'chiller_feature_cols.pkl')

# Save predictions CSV
pred_df = pd.DataFrame({
    'Timestamp':       pd.date_range(start="2024-01-01", 
                                     periods=len(y_test), freq="1min"),
    'Actual_Label':    y_test,
    'Predicted_Label': y_pred_best,
    'Actual_Class':    [LABEL_MAP[l] for l in y_test],
    'Predicted_Class': [LABEL_MAP[l] for l in y_pred_best],
})
for i, name in LABEL_MAP.items():
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(X_test_s)
        pred_df[f'Prob_{name}'] = proba[:, i] if i < proba.shape[1] else 0.0

pred_df.to_csv(RUN_DIR / 'chiller_predictions.csv', index=False)
pred_df.to_csv(BASE_DIR / 'data' / 'chiller_predictions.csv', index=False)

# Feature sample for dashboard
feat_sample = pd.DataFrame(X_test_s, columns=feat_cols)
feat_sample['Actual_Label']    = y_test
feat_sample['Predicted_Label'] = y_pred_best
feat_sample.to_csv(BASE_DIR / 'data' / 'chiller_feature_sample.csv', index=False)

# Summary JSON
summary = {
    'dataset':        'CHILLER_RP1043',
    'run_timestamp':  TS,
    'n_train':        int(len(X_train)),
    'n_test':         int(len(X_test)),
    'n_features':     int(len(feat_cols)),
    'macro_f1':       round(best_f1, 4),
    'ci_lo':          ci_lo,
    'ci_hi':          ci_hi,
    'model_results':  results,
    'label_map':      LABEL_MAP,
}
with open(RUN_DIR / 'chiller_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  ✓ Model saved     → models/chiller_rf_model.pkl")
print(f"  ✓ Predictions     → data/chiller_predictions.csv")
print(f"  ✓ Feature sample  → data/chiller_feature_sample.csv")
print(f"  ✓ Summary JSON    → {RUN_DIR.name}/chiller_results_summary.json")

print("\n" + "=" * 65)
print("  RESULTS — CHILLER_RP1043")
print("=" * 65)
print(f"  Macro-F1 (in-sample) : {best_f1:.4f}  95% CI [{ci_lo}, {ci_hi}]")
print(f"  Best model           : {best_name}")
print(f"  Features used        : {len(feat_cols)}")
print(f"  Run saved to         : results/run_{TS}_CHILLER_RP1043/")
print("=" * 65)
