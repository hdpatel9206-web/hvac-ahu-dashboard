"""
╔══════════════════════════════════════════════════════════════╗
║  HVAC FAULT DETECTION — UNIVERSAL EXPERIMENT RUNNER         ║
║  Works with ANY dataset — just change the config file       ║
║                                                              ║
║  Usage:                                                      ║
║    python pipeline/run_experiment.py --config configs/config_ashrae.py║
║    python pipeline/run_experiment.py --config configs/config_new.py  ║
║                                                              ║
║  Each run saves to its own timestamped folder in results/   ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import importlib.util
import os
import sys
import json
import pathlib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, classification_report, precision_score, recall_score
)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════
# PARSE ARGUMENT
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='HVAC Fault Detection Experiment Runner')
parser.add_argument('--config', type=str, required=True,
                    help='Path to config file, e.g. configs/config_ashrae.py')
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════
# LOAD CONFIG DYNAMICALLY
# ══════════════════════════════════════════════════════════════
config_path = pathlib.Path(args.config).resolve()
if not config_path.exists():
    print(f"\n  ERROR: Config file not found: {config_path}")
    print(f"  Create a config file first. See configs/config_ashrae.py as example.\n")
    sys.exit(1)

spec = importlib.util.spec_from_file_location('config', config_path)
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# ── Resolve all paths relative to thesis root ─────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / cfg.DATA_DIR

# Results folder: results/run_YYYYMMDD_HHMMSS_DATASETNAME/
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name   = f"run_{timestamp}_{cfg.DATASET_NAME}"
RUN_DIR    = BASE_DIR / 'results' / run_name
MODEL_DIR  = RUN_DIR / 'models'
FIG_DIR    = RUN_DIR / 'figures'

for d in [RUN_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Also update main models/ and data/ folders for dashboard
MAIN_MODEL_DIR = BASE_DIR / 'models'
MAIN_DATA_DIR  = BASE_DIR / 'data'
MAIN_MODEL_DIR.mkdir(exist_ok=True)

RS = getattr(cfg, 'RANDOM_STATE', 42)

print("=" * 65)
print(f"  HVAC EXPERIMENT RUNNER")
print(f"  Dataset   : {cfg.DATASET_NAME}")
print(f"  Run folder: results/{run_name}")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print(f"\n[1/6] Loading data from {DATA_DIR}...")

def load_site(filename, fault_class, datetime_col='Datetime'):
    """Load one CSV file and assign a fault class label."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"\n  ERROR: {path} not found.\n"
            f"  Check DATA_DIR in your config file.\n"
        )
    # Try to detect datetime column
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Rename datetime column to Timestamp
    dt_candidates = [datetime_col, 'Datetime', 'Timestamp', 'Time', 'DATE', 'timestamp']
    for cand in dt_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: 'Timestamp'})
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            break
    else:
        # No datetime column found — create a sequential index
        df['Timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='1min')

    # Force all feature columns to numeric
    for c in df.columns:
        if c not in ['Timestamp', 'site', 'label']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.ffill().fillna(0)
    df['site'] = filename.replace('.csv', '')

    # Assign label from ground truth column
    gt_col = cfg.GROUND_TRUTH_COL
    if gt_col not in df.columns:
        raise ValueError(
            f"\n  ERROR: Ground truth column '{gt_col}' not found in {filename}.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  Update GROUND_TRUTH_COL in your config file.\n"
        )
    fault_val = getattr(cfg, 'FAULT_VALUE', 1)
    df['label'] = df[gt_col].apply(lambda x: fault_class if x == fault_val else 0)

    return df

# Load training sites
all_train_dfs = []
for filename, fault_class in cfg.TRAIN_FILES.items():
    df = load_site(filename, fault_class)
    cb_flag = ''
    print(f"  TRAIN  {filename:<22}: {len(df):>7,} rows | labels={dict(df['label'].value_counts().sort_index())}{cb_flag}")
    all_train_dfs.append(df)

# Load cross-building sites
all_cross_dfs = []
for filename, fault_class in cfg.CROSS_BUILDING_FILES.items():
    df = load_site(filename, fault_class)
    print(f"  CROSS  {filename:<22}: {len(df):>7,} rows | labels={dict(df['label'].value_counts().sort_index())}  [CROSS-BLDG]")
    all_cross_dfs.append(df)

# ══════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print(f"\n[2/6] Engineering features...")

def engineer(df, feature_cols, windows):
    df = df.copy().sort_values('Timestamp').reset_index(drop=True)
    for col in feature_cols:
        if col in df.columns:
            for w in windows:
                df[f'{col}_rm{w}'] = df[col].rolling(w, min_periods=1).mean()
                df[f'{col}_rs{w}'] = df[col].rolling(w, min_periods=1).std().fillna(0)
    # Derived features from config
    for derived_name, (col_a, col_b) in getattr(cfg, 'DERIVED_FEATURES', {}).items():
        if col_a in df.columns and col_b in df.columns:
            df[derived_name] = df[col_a] - df[col_b]
    return df

WINDOWS = getattr(cfg, 'ROLLING_WINDOWS', [10, 30])
FEAT_COLS = cfg.FEATURE_COLS

engineered_train = [engineer(df, FEAT_COLS, WINDOWS) for df in all_train_dfs]
engineered_cross = [engineer(df, FEAT_COLS, WINDOWS) for df in all_cross_dfs]

# Find columns common to all dataframes
EXCLUDE = set(getattr(cfg, 'EXCLUDE_COLS', [])) | {
    'Timestamp', 'site', 'label'
}
EXCLUDE.add(cfg.GROUND_TRUTH_COL)

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

final_feat_cols = get_feat_cols(engineered_train[0])
for df in engineered_train[1:] + engineered_cross:
    final_feat_cols = [c for c in final_feat_cols if c in df.columns]

print(f"  Feature columns: {len(final_feat_cols)}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — BUILD TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
print(f"\n[3/6] Building train/test split...")

# Apply subsampling if configured (to prevent large sites dominating)
SUBSAMPLE = getattr(cfg, 'SUBSAMPLE_PER_CLASS', None)

sampled_dfs = []
for df in engineered_train:
    if SUBSAMPLE:
        parts = []
        for lbl in df['label'].unique():
            subset = df[df['label'] == lbl]
            n = min(len(subset), SUBSAMPLE)
            parts.append(subset.sample(n=n, random_state=RS))
        sampled_dfs.append(pd.concat(parts))
    else:
        sampled_dfs.append(df)

# Per-building Z-score normalization before merging
print(f"  Applying per-building normalization...")
scalers = {}
normed_train_dfs = []
for i, df in enumerate(sampled_dfs):
    scaler = StandardScaler()
    X_normed = scaler.fit_transform(df[final_feat_cols])
    scalers[f'train_{i}'] = scaler
    normed_df = df.copy()
    normed_df[final_feat_cols] = X_normed
    normed_train_dfs.append(normed_df)

# Normalize cross-building data with separate scalers
normed_cross_dfs = []
for i, df in enumerate(engineered_cross):
    scaler = StandardScaler()
    X_normed = scaler.fit_transform(df[final_feat_cols])
    scalers[f'cross_{i}'] = scaler
    normed_df = df.copy()
    normed_df[final_feat_cols] = X_normed
    normed_cross_dfs.append(normed_df)

train_df = pd.concat(normed_train_dfs).sample(frac=1, random_state=RS).reset_index(drop=True)
X_all = train_df[final_feat_cols].values
y_all = train_df['label'].values

TEST_SIZE = getattr(cfg, 'TEST_SIZE', 0.20)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    stratify=y_all,
    random_state=RS
)

# Cross-building data (already normalized per-building)
cb_df   = pd.concat(normed_cross_dfs).reset_index(drop=True)
X_cross = cb_df[final_feat_cols].values
y_cross = cb_df['label'].values

# No global scaling needed - data is already normalized per-building
X_tr_sc = X_train
X_te_sc = X_test
X_cb_sc = X_cross

print(f"  Training samples  : {len(X_train):,}")
print(f"  Test samples      : {len(X_test):,}")
print(f"  Cross-bldg samples: {len(X_cross):,}")
print(f"  Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print(f"\n[4/6] Training models...")

LABEL_MAP = getattr(cfg, 'LABEL_MAP', {
    0: 'Healthy', 1: 'Fault-1', 2: 'Fault-2', 3: 'Fault-3'
})

models_to_train = {
    'Rule-based (Dummy)': DummyClassifier(strategy='stratified', random_state=RS),
    'SVM':                SVC(kernel='rbf', C=1.0, class_weight='balanced',
                              probability=True, random_state=RS),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                      random_state=RS),
    'Random Forest':      RandomForestClassifier(
                            n_estimators=10, max_features='sqrt',
                            class_weight='balanced', n_jobs=-1, random_state=RS),
}

# Add Gradient Boosting if configured
if not getattr(cfg, 'INCLUDE_GRADIENT_BOOSTING', True):
    models_to_train.pop('Gradient Boosting', None)

trained = {}
print(f"\n  {'Model':<26} {'Macro-F1':>10} {'MCC':>8} {'Bal-Acc':>10}")
print(f"  {'-'*26} {'-'*10} {'-'*8} {'-'*10}")

for name, model in models_to_train.items():
    model.fit(X_tr_sc, y_train)
    yp = model.predict(X_te_sc)
    f1  = f1_score(y_test, yp, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, yp)
    bal = balanced_accuracy_score(y_test, yp)
    trained[name] = dict(model=model, yp=yp, f1=f1, mcc=mcc, bal=bal)
    marker = ' ← BEST' if name == 'Random Forest' else ''
    print(f"  {name:<26} {f1:>10.4f} {mcc:>8.4f} {bal:>10.4f}{marker}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — METRICS + BOOTSTRAP CI + CROSS-BUILDING
# ══════════════════════════════════════════════════════════════
print(f"\n[5/6] Computing metrics and cross-building evaluation...")

rf     = trained['Random Forest']['model']
rf_pred = trained['Random Forest']['yp']
pres   = sorted(np.unique(np.concatenate([y_test, rf_pred])))
tnames = [LABEL_MAP.get(l, str(l)) for l in pres]

# Classification report
print(f"\n  === Random Forest — Classification Report ===")
print(classification_report(y_test, rf_pred, labels=pres,
                             target_names=tnames, zero_division=0))

# FPR per class
cm = confusion_matrix(y_test, rf_pred, labels=pres)
print(f"  False Positive Rate per class:")
fpr_dict = {}
for i, lbl in enumerate(pres):
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fpr_dict[LABEL_MAP.get(lbl, str(lbl))] = round(float(fpr), 4)
    print(f"    {LABEL_MAP.get(lbl, str(lbl)):<28}: {fpr:.4f}  ({fpr*100:.1f}%)")

# Bootstrap CI
N_BOOT = getattr(cfg, 'N_BOOTSTRAP', 300)
np.random.seed(RS)
boot_f1, boot_mcc = [], []
for _ in range(N_BOOT):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test[idx])) > 1:
        boot_f1.append(f1_score(y_test[idx], rf_pred[idx], average='macro', zero_division=0))
        boot_mcc.append(matthews_corrcoef(y_test[idx], rf_pred[idx]))

f1_mean = np.mean(boot_f1)
f1_lo   = np.percentile(boot_f1, 2.5)
f1_hi   = np.percentile(boot_f1, 97.5)
mcc_mean = np.mean(boot_mcc)

print(f"\n  Macro-F1  : {f1_mean:.4f}  95% CI [{f1_lo:.4f}, {f1_hi:.4f}]")
print(f"  MCC       : {mcc_mean:.4f}")

# Cross-building
cross_pred = rf.predict(X_cb_sc)
cb_pres    = sorted(np.unique(np.concatenate([y_cross, cross_pred])))
f1_cb      = f1_score(y_cross, cross_pred, average='macro', zero_division=0, labels=cb_pres)
mcc_cb     = matthews_corrcoef(y_cross, cross_pred)
bal_cb     = balanced_accuracy_score(y_cross, cross_pred)
f1_in      = trained['Random Forest']['f1']
drop       = f1_in - f1_cb

print(f"\n  Cross-building Macro-F1 : {f1_cb:.4f}")
print(f"  Generalisation drop     : {drop:.4f}  ({drop*100:.1f} pp)")

# PSI — distribution shift
def psi(a, b, bins=10):
    mn, mx = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(mn, mx, bins + 1)
    ea = np.histogram(a, edges)[0] / len(a) + 1e-6
    eb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((ea - eb) * np.log(ea / eb)))

print(f"\n  PSI — top feature distribution shifts:")
psi_results = {}
for kf in final_feat_cols[:8]:
    idx = final_feat_cols.index(kf)
    p_val = psi(X_tr_sc[:, idx], X_cb_sc[:, idx])
    flag = '⚠  HIGH' if p_val > 0.2 else ('⚡ MOD' if p_val > 0.1 else '✓ OK')
    psi_results[kf] = round(p_val, 4)
    print(f"    {kf[:40]:<40} PSI={p_val:.4f}  {flag}")

summary = {}

# ── McNemar's Test — RF vs each baseline ──────────────────────
print(f"\n  McNemar's Test (RF vs each baseline):")
print(f"  {'Comparison':<35} {'chi2':>8} {'p-value':>10} {'Result':>16}")
print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*16}")

try:
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
    rf_correct = (trained['Random Forest']['yp'] == y_test)
    mcnemar_results = {}

    for name in trained:
        if name == 'Random Forest':
            continue
        other_correct = (trained[name]['yp'] == y_test)
        n00 = int(((~rf_correct) & (~other_correct)).sum())
        n01 = int(((~rf_correct) & ( other_correct)).sum())
        n10 = int((rf_correct & (~other_correct)).sum())
        n11 = int((rf_correct & other_correct).sum())
        table = [[n11, n10], [n01, n00]]
        result = mcnemar_test(table, exact=False, correction=True)
        sig = 'SIGNIFICANT ✓' if result.pvalue < 0.05 else 'not significant'
        mcnemar_results[name] = {
            'chi2': round(float(result.statistic), 3),
            'pvalue': round(float(result.pvalue), 6),
            'significant': bool(result.pvalue < 0.05)
        }
        print(f"  RF vs {name:<28} {result.statistic:>8.3f} {result.pvalue:>10.4f} {sig:>16}")

    summary['mcnemar_tests'] = mcnemar_results

except Exception as e:
    print(f"  McNemar test error: {e}")

# ══════════════════════════════════════════════════════════════
# SAVE MODELS + RESULTS + CSV
# ══════════════════════════════════════════════════════════════
# Save to run folder
joblib.dump(rf,             MODEL_DIR / 'rf_model.pkl')
joblib.dump(final_feat_cols, MODEL_DIR / 'feature_cols.pkl')
joblib.dump(scalers,        MODEL_DIR / 'scalers.pkl')

# Also overwrite main models/ so dashboard always uses latest
joblib.dump(rf,             MAIN_MODEL_DIR / 'rf_model.pkl')
joblib.dump(final_feat_cols, MAIN_MODEL_DIR / 'feature_cols.pkl')
joblib.dump(scalers,        MAIN_MODEL_DIR / 'scalers.pkl')

# Save predictions CSV
proba   = rf.predict_proba(X_cb_sc)
prob_df = pd.DataFrame(proba,
    columns=[f'Prob_{LABEL_MAP.get(c, str(c))}' for c in rf.classes_])
pred_df = pd.DataFrame({
    'Timestamp':       cb_df['Timestamp'].values,
    'Actual_Label':    y_cross,
    'Predicted_Label': cross_pred,
    'Actual_Class':    [LABEL_MAP.get(l, '?') for l in y_cross],
    'Predicted_Class': [LABEL_MAP.get(l, '?') for l in cross_pred],
})
pred_df = pd.concat([pred_df, prob_df], axis=1)
pred_df.to_csv(RUN_DIR / 'chiller_predictions.csv', index=False)
pred_df.to_csv(MAIN_DATA_DIR / 'chiller_predictions.csv', index=False)

# Save feature sample for dashboard SHAP and live prediction
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
feature_sample = X_test_df.copy()
feature_sample['Actual_Label'] = y_test
feature_sample['Predicted_Label'] = y_pred
feature_sample.to_csv(RUN_DIR / 'feature_sample.csv', index=False)
feature_sample.to_csv(BASE_DIR / 'data' / 'feature_sample.csv', index=False)
print(f"  ✓ Feature sample → {BASE_DIR / 'data' / 'feature_sample.csv'}")

# Save JSON summary
summary = {
    'dataset':          cfg.DATASET_NAME,
    'run_timestamp':    timestamp,
    'n_train':          int(len(X_train)),
    'n_test':           int(len(X_test)),
    'n_cross_building': int(len(X_cross)),
    'n_features':       len(final_feat_cols),
    'macro_f1_insample': round(float(f1_mean), 4),
    'macro_f1_ci_lo':   round(float(f1_lo), 4),
    'macro_f1_ci_hi':   round(float(f1_hi), 4),
    'mcc':              round(float(mcc_mean), 4),
    'balanced_acc':     round(float(trained['Random Forest']['bal']), 4),
    'f1_cross_building': round(float(f1_cb), 4),
    'generalisation_drop_pp': round(float(drop * 100), 2),
    'fpr_per_class':    fpr_dict,
    'psi_top_features': psi_results,
    'mcnemar_tests':    summary.get('mcnemar_tests', {}),
    'model_comparison': {
        name: {'f1': round(v['f1'], 4), 'mcc': round(v['mcc'], 4), 'bal': round(v['bal'], 4)}
        for name, v in trained.items()
    }
}
with open(RUN_DIR / 'results_summary.json', 'w') as f:
    json.dump(
        summary,
        f,
        indent=2,
        default=lambda x: bool(x) if hasattr(x, '__bool__') and not isinstance(x, (int, float, str, list, dict)) else str(x)
    )

print(f"\n  ✓ Models saved  → {MODEL_DIR}")
print(f"  ✓ Models copied → {MAIN_MODEL_DIR}  (dashboard updated)")
print(f"  ✓ Predictions   → {RUN_DIR / 'chiller_predictions.csv'}")
print(f"  ✓ Summary JSON  → {RUN_DIR / 'results_summary.json'}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — FIGURES
# ══════════════════════════════════════════════════════════════
print(f"\n[6/6] Generating figures...")

mnames = list(trained.keys())
bcols  = ['#95a5a6', '#e67e22', '#27ae60'][:len(mnames)]

# Fig 1 — model comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, (vals, title) in zip(axes, [
    ([trained[m]['f1']  for m in mnames], 'Macro F1-Score'),
    ([trained[m]['mcc'] for m in mnames], 'Matthews Corr. Coeff.'),
    ([trained[m]['bal'] for m in mnames], 'Balanced Accuracy'),
]):
    bars = ax.bar(mnames, vals, color=bcols[:len(mnames)], edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_xticklabels(mnames, rotation=12, ha='right', fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    bars[-1].set_edgecolor('#1a3a5c')
    bars[-1].set_linewidth(2.5)
plt.suptitle(f'Model Comparison — {cfg.DATASET_NAME}', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2 — confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (yt, yp2, title) in zip(axes, [
    (y_test, rf_pred, 'In-Sample Test'),
    (y_cross, cross_pred, 'Cross-Building Test'),
]):
    pres2   = sorted(np.unique(np.concatenate([yt, yp2])))
    tnames2 = [LABEL_MAP.get(l, '?') for l in pres2]
    cmx     = confusion_matrix(yt, yp2, labels=pres2).astype(float)
    rs      = cmx.sum(axis=1, keepdims=True)
    cmx_pct = np.divide(cmx, rs, where=rs != 0)
    sns.heatmap(cmx_pct, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tnames2, yticklabels=tnames2,
                ax=ax, linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {title}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3 — per-class F1
fig, ax = plt.subplots(figsize=(10, 5))
rf_pf = f1_score(y_test, rf_pred, labels=pres, average=None, zero_division=0)
bar_colors = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad', '#2980b9']
ax.bar(range(len(pres)), rf_pf,
       color=bar_colors[:len(pres)], edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(pres)))
ax.set_xticklabels(tnames, fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title(f'Per-Class F1-Score — Random Forest\n{cfg.DATASET_NAME}',
             fontweight='bold', fontsize=12)
for i, v in enumerate(rf_pf):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_perclass_f1.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4 — cross-building degradation
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([0], [f1_in], 0.4, color='#27ae60', label=f'In-Sample ({f1_in:.3f})')
ax.bar([1], [f1_cb], 0.4, color='#e74c3c', label=f'Cross-Building ({f1_cb:.3f})')
ax.annotate('', xy=(1, f1_cb), xytext=(0, f1_in),
            arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=2.5))
ax.text(0.5, (f1_in + f1_cb) / 2 + 0.03,
        f'−{drop*100:.1f} pp', ha='center', fontsize=13,
        color='#c0392b', fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['In-Sample\nTest', 'Cross-Building\nTest'], fontsize=11)
ax.set_ylabel('Macro F1-Score', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title(f'Cross-Building Generalisation\n{cfg.DATASET_NAME}',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
for i, v in enumerate([f1_in, f1_cb]):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_cross_building.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 5 — feature importance
fig, ax = plt.subplots(figsize=(10, 6))
imp   = rf.feature_importances_
top_i = np.argsort(imp)[-15:]
top_v = imp[top_i]
top_n = [final_feat_cols[i][:42] for i in top_i]
bar_c = ['#e74c3c' if v > np.percentile(top_v, 75) else
         '#e67e22' if v > np.percentile(top_v, 50) else '#3498db'
         for v in top_v]
ax.barh(range(len(top_i)), top_v, color=bar_c, edgecolor='white', linewidth=0.8)
ax.set_yticks(range(len(top_i)))
ax.set_yticklabels(top_n, fontsize=9)
ax.set_xlabel('Gini Feature Importance', fontsize=10)
ax.set_title(f'Top Feature Importances — Random Forest\n{cfg.DATASET_NAME}',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ 5 figures saved → {FIG_DIR}")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  RESULTS — {cfg.DATASET_NAME}")
print("=" * 65)
print(f"  Macro-F1 (in-sample) : {f1_mean:.4f}  95% CI [{f1_lo:.4f}, {f1_hi:.4f}]")
print(f"  MCC                  : {mcc_mean:.4f}")
print(f"  Balanced Accuracy    : {trained['Random Forest']['bal']:.4f}")
print(f"  Cross-building F1    : {f1_cb:.4f}")
print(f"  Generalisation drop  : {drop*100:.1f} percentage points")
print(f"\n  Run saved to : results/{run_name}/")
print(f"  Dashboard    : streamlit run hvac_dashboard/app.py")
print("=" * 65)
