"""
HVAC Multi-Fault Detection Pipeline
Run this once to generate:
  - models/rf_model.pkl
  - models/feature_cols.pkl
  - models/scaler.pkl
  - data/chiller_predictions.csv
  - data/figures/fig1-fig5.png
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, classification_report
)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ════════════════════════════════════════════════════════════════
# PATHS — adjust if your folder is not on Desktop
# ════════════════════════════════════════════════════════════════
import pathlib

# Automatically detect the folder this script is in
BASE_DIR   = pathlib.Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / 'data'          # where your CSVs are
MODEL_DIR  = BASE_DIR / 'models'        # where pkl files go
OUTPUT_DIR = BASE_DIR / 'data' / 'figures'  # where figures go

# Create folders if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  HVAC PIPELINE — REAL ASHRAE DATA")
print(f"  Data   : {DATA_DIR}")
print(f"  Models : {MODEL_DIR}")
print(f"  Figures: {OUTPUT_DIR}")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
RS = 42
LABEL_MAP = {
    0: 'Healthy',
    1: 'Fault-A (OA Sensor)',
    2: 'Fault-B (Valve Leak)',
    3: 'Fault-C (Control)',
}
FEATS = [
    'AHU: Supply Air Temperature',
    'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',
    'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',
    'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Cooling Coil Valve Control Signal',
    'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════
print("\n[1/6] Loading ASHRAE datasets...")

def load_site(filename, fault_class):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"\n  ERROR: Cannot find {path}\n"
            f"  Make sure you extracted 910.zip and the inner zip into the data/ folder."
        )
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.rename(columns={'Datetime': 'Timestamp'})
    df.columns = [c.strip() for c in df.columns]
    # Force numeric (some files store numbers as strings)
    for c in FEATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.ffill().fillna(0)
    df['label'] = df['Fault Detection Ground Truth'].apply(
        lambda x: fault_class if x == 1 else 0
    )
    return df

mzvav1   = load_site('MZVAV-1.csv',   fault_class=1)
mzvav2_1 = load_site('MZVAV-2-1.csv', fault_class=2)
szcav    = load_site('SZCAV.csv',     fault_class=3)
mzvav2_2 = load_site('MZVAV-2-2.csv', fault_class=2)  # cross-building test
szvav    = load_site('SZVAV.csv',     fault_class=3)   # cross-building test

for name, df in [('MZVAV-1', mzvav1), ('MZVAV-2-1', mzvav2_1), ('SZCAV', szcav),
                 ('MZVAV-2-2', mzvav2_2), ('SZVAV', szvav)]:
    cb = ' [CROSS-BUILDING]' if name in ['MZVAV-2-2', 'SZVAV'] else ''
    label_dist = dict(df['label'].value_counts().sort_index())
    print(f"  {name:<12}: {len(df):>7,} rows | labels={label_dist}{cb}")

# ════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
print("\n[2/6] Engineering features (rolling windows 10 & 30)...")

def engineer(df):
    df = df.copy().sort_values('Timestamp').reset_index(drop=True)
    for col in FEATS:
        if col in df.columns:
            for w in [10, 30]:
                df[f'{col}_rm{w}'] = df[col].rolling(w, min_periods=1).mean()
                df[f'{col}_rs{w}'] = df[col].rolling(w, min_periods=1).std().fillna(0)
    # Derived features
    if 'AHU: Supply Air Temperature' in df.columns and 'AHU: Return Air Temperature' in df.columns:
        df['Temp_supply_return_diff'] = (
            df['AHU: Supply Air Temperature'] - df['AHU: Return Air Temperature']
        )
        df['Temp_outdoor_supply_diff'] = (
            df['AHU: Supply Air Temperature'] - df['AHU: Outdoor Air Temperature']
        )
    return df

mzvav1   = engineer(mzvav1)
mzvav2_1 = engineer(mzvav2_1)
szcav    = engineer(szcav)
mzvav2_2 = engineer(mzvav2_2)
szvav    = engineer(szvav)

# Build feature column list — only columns common to all sites
EXCLUDE = {
    'Timestamp', 'site', 'label', 'Fault Detection Ground Truth',
    'AHU: Supply Air Temperature Set Point',
    'AHU: Supply Air Temperature Heating Set Point',
    'AHU: Supply Air Temperature Cooling Set Point',
    'AHU: Supply Air Duct Static Pressure Set Point',
    'AHU: Supply Air Duct Static Pressure',
    'AHU: Return Air Fan Status',
    'AHU: Return Air Fan Speed Control Signal',
    'AHU: Exhaust Air Damper Control Signal',
}

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

feat_cols = get_feature_cols(mzvav1)
for df in [mzvav2_1, mzvav2_2, szcav, szvav]:
    feat_cols = [c for c in feat_cols if c in df.columns]

print(f"  Total feature columns: {len(feat_cols)}")

# ════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════════
print("\n[3/6] Building train/test split...")

# Subsample MZVAV-1 (272k rows) to avoid imbalance
healthy = mzvav1[mzvav1['label'] == 0].sample(n=8000, random_state=RS)
faulty  = mzvav1[mzvav1['label'] == 1].sample(n=8000, random_state=RS)

train_df = pd.concat([healthy, faulty, mzvav2_1, szcav]).sample(
    frac=1, random_state=RS
).reset_index(drop=True)

X_all = train_df[feat_cols].values
y_all = train_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.20, stratify=y_all, random_state=RS
)

# Cross-building data (never seen during training)
cb_df   = pd.concat([mzvav2_2, szvav]).reset_index(drop=True)
X_cross = cb_df[feat_cols].values
y_cross = cb_df['label'].values

print(f"  Training   : {len(X_train):,} samples")
print(f"  Test       : {len(X_test):,} samples")
print(f"  Cross-bldg : {len(X_cross):,} samples")
print(f"  Label dist (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

# Scale
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)
X_cb_sc  = scaler.transform(X_cross)

# ════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN MODEL
# ════════════════════════════════════════════════════════════════
print("\n[4/6] Training Random Forest (this takes ~2 minutes)...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=RS
)
rf.fit(X_tr_sc, y_train)

rf_pred    = rf.predict(X_te_sc)
cross_pred = rf.predict(X_cb_sc)

# ── Metrics ──────────────────────────────────────────────────────
pres   = sorted(np.unique(np.concatenate([y_test, rf_pred])))
tnames = [LABEL_MAP.get(l, str(l)) for l in pres]

f1_in  = f1_score(y_test,  rf_pred,    average='macro', zero_division=0)
f1_cb  = f1_score(y_cross, cross_pred, average='macro', zero_division=0,
                  labels=sorted(np.unique(np.concatenate([y_cross, cross_pred]))))
mcc    = matthews_corrcoef(y_test, rf_pred)
bal    = balanced_accuracy_score(y_test, rf_pred)
drop   = f1_in - f1_cb

# Bootstrap 95% CI
np.random.seed(RS)
boot = []
for _ in range(300):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test[idx])) > 1:
        boot.append(f1_score(y_test[idx], rf_pred[idx], average='macro', zero_division=0))
ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)

# Confusion matrix FPR
cm = confusion_matrix(y_test, rf_pred, labels=pres)

print(f"\n  ╔══════════════════════════════════════════╗")
print(f"  ║  RESULTS SUMMARY                         ║")
print(f"  ╠══════════════════════════════════════════╣")
print(f"  ║  Macro F1     : {f1_in:.4f}                    ║")
print(f"  ║  95% CI       : [{ci_lo:.4f}, {ci_hi:.4f}]          ║")
print(f"  ║  MCC          : {mcc:.4f}                    ║")
print(f"  ║  Balanced Acc : {bal:.4f}                    ║")
print(f"  ╠══════════════════════════════════════════╣")
print(f"  ║  Cross-bldg F1: {f1_cb:.4f}                    ║")
print(f"  ║  Drop         : {drop*100:.1f} pp                      ║")
print(f"  ╚══════════════════════════════════════════╝")
print()
print(classification_report(y_test, rf_pred, labels=pres,
                             target_names=tnames, zero_division=0))

print("  False Positive Rate per class:")
for i, lbl in enumerate(pres):
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"    {LABEL_MAP.get(lbl, str(lbl)):<28}: {fpr:.4f}  ({fpr*100:.1f}%)")

# ════════════════════════════════════════════════════════════════
# STEP 5 — SAVE MODELS + PREDICTIONS CSV
# ════════════════════════════════════════════════════════════════
print("\n[5/6] Saving models and predictions...")

joblib.dump(rf,        MODEL_DIR / 'rf_model.pkl')
joblib.dump(feat_cols, MODEL_DIR / 'feature_cols.pkl')
joblib.dump(scaler,    MODEL_DIR / 'scaler.pkl')
print(f"  ✓ rf_model.pkl       → {MODEL_DIR}")
print(f"  ✓ feature_cols.pkl   → {MODEL_DIR}")
print(f"  ✓ scaler.pkl         → {MODEL_DIR}")

# Predictions CSV (for dashboard)
proba = rf.predict_proba(X_cb_sc)
prob_df = pd.DataFrame(proba, columns=[f'Prob_{LABEL_MAP.get(c, str(c))}' for c in rf.classes_])

pred_df = pd.DataFrame({
    'Timestamp':       cb_df['Timestamp'].values,
    'Actual_Label':    y_cross,
    'Predicted_Label': cross_pred,
    'Actual_Class':    [LABEL_MAP.get(l, '?') for l in y_cross],
    'Predicted_Class': [LABEL_MAP.get(l, '?') for l in cross_pred],
})
pred_df = pd.concat([pred_df, prob_df], axis=1)
pred_df.to_csv(DATA_DIR / 'chiller_predictions.csv', index=False)
print(f"  ✓ chiller_predictions.csv → {DATA_DIR}")

# ════════════════════════════════════════════════════════════════
# STEP 6 — GENERATE FIGURES
# ════════════════════════════════════════════════════════════════
print("\n[6/6] Generating thesis figures...")

# ── Fig 1: Model comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
dummy_pred = DummyClassifier(strategy='stratified', random_state=RS).fit(X_tr_sc, y_train).predict(X_te_sc)
mnames = ['Rule-based', 'Random Forest']
f1s  = [f1_score(y_test, dummy_pred, average='macro', zero_division=0), f1_in]
mccs = [matthews_corrcoef(y_test, dummy_pred), mcc]
bals = [balanced_accuracy_score(y_test, dummy_pred), bal]
bcols = ['#95a5a6', '#27ae60']
for ax, (vals, title) in zip(axes, [(f1s, 'Macro F1-Score'), (mccs, 'Matthews Corr. Coeff.'), (bals, 'Balanced Accuracy')]):
    bars = ax.bar(mnames, vals, color=bcols, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(mnames, rotation=10, ha='right', fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    bars[-1].set_edgecolor('#1a3a5c')
    bars[-1].set_linewidth(2.5)
plt.suptitle('Model Comparison — Real ASHRAE LBNL Dataset', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig1_model_comparison.png")

# ── Fig 2: Confusion matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (yt, yp, title) in zip(axes, [
    (y_test, rf_pred, 'In-Sample Test Set'),
    (y_cross, cross_pred, 'Cross-Building Test'),
]):
    pres2  = sorted(np.unique(np.concatenate([yt, yp])))
    tnames2 = [LABEL_MAP.get(l, '?') for l in pres2]
    cmx = confusion_matrix(yt, yp, labels=pres2).astype(float)
    rs  = cmx.sum(axis=1, keepdims=True)
    cmx_pct = np.divide(cmx, rs, where=rs != 0)
    sns.heatmap(cmx_pct, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tnames2, yticklabels=tnames2, ax=ax, linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {title}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig2_confusion_matrix.png")

# ── Fig 3: Per-class F1 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
rf_pf = f1_score(y_test, rf_pred, labels=pres, average=None, zero_division=0)
bar_colors = ['#27ae60', '#e74c3c', '#e67e22', '#8e44ad']
bars = ax.bar(range(len(pres)), rf_pf,
              color=bar_colors[:len(pres)], edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(pres)))
ax.set_xticklabels(tnames, fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title('Per-Class F1-Score — Random Forest\n(Real ASHRAE Data, In-Sample Test)',
             fontweight='bold', fontsize=12)
for i, v in enumerate(rf_pf):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_perclass_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig3_perclass_f1.png")

# ── Fig 4: Cross-building degradation ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([0], [f1_in], 0.4, color='#27ae60', label=f'In-Sample ({f1_in:.3f})')
ax.bar([1], [f1_cb], 0.4, color='#e74c3c', label=f'Cross-Building ({f1_cb:.3f})')
ax.annotate('', xy=(1, f1_cb), xytext=(0, f1_in),
            arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=2.5))
ax.text(0.5, (f1_in + f1_cb) / 2 + 0.03,
        f'−{drop*100:.1f} pp', ha='center', fontsize=13,
        color='#c0392b', fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['In-Sample Test', 'Cross-Building\nTest'], fontsize=11)
ax.set_ylabel('Macro F1-Score', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Cross-Building Generalisation\nRandom Forest — Real ASHRAE Data',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
for i, v in enumerate([f1_in, f1_cb]):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_cross_building.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig4_cross_building.png")

# ── Fig 5: Feature importance ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
imp    = rf.feature_importances_
top_i  = np.argsort(imp)[-15:]
top_v  = imp[top_i]
top_n  = [feat_cols[i][:42] for i in top_i]
bar_c  = ['#e74c3c' if v > np.percentile(top_v, 75) else
          '#e67e22' if v > np.percentile(top_v, 50) else '#3498db' for v in top_v]
ax.barh(range(15), top_v, color=bar_c, edgecolor='white', linewidth=0.8)
ax.set_yticks(range(15))
ax.set_yticklabels(top_n, fontsize=9)
ax.set_xlabel('Gini Feature Importance', fontsize=10)
ax.set_title('Top 15 Feature Importances — Random Forest\n(Real ASHRAE Data)',
             fontweight='bold', fontsize=11)
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color='#e74c3c', label='Top 25%'),
    mpatches.Patch(color='#e67e22', label='Top 50%'),
    mpatches.Patch(color='#3498db', label='Other'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig5_feature_importance.png")

# ════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ALL DONE")
print("=" * 60)
print(f"\n  Macro-F1  : {f1_in:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  MCC       : {mcc:.4f}")
print(f"  Cross-bldg: {f1_cb:.4f}  (drop = {drop*100:.1f} pp)")
print(f"\n  Models saved  → {MODEL_DIR}")
print(f"  CSV saved     → {DATA_DIR / 'chiller_predictions.csv'}")
print(f"  Figures saved → {OUTPUT_DIR}")
print("\n  Next step: run  streamlit run hvac_dashboard/app.py")
print("=" * 60)
