# =============================================================================
# THESIS REPRODUCTION SCRIPT
# Multi-Fault Detection for Smart Building AHUs
# Harshil Patel - bbw Hochschule Berlin - 2026
#
# Run: python thesis_reproduce.py
# Expected runtime: 15-20 minutes
# =============================================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import f1_score, classification_report
from sklearn.model_selection import train_test_split

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
TRAIN_FILES  = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]
TEST_FILES   = ["MZVAV-2-2.csv", "SZVAV.csv"]
FAULT_MAP    = {0:"Healthy", 1:"Sensor Bias", 2:"Valve Fault", 3:"Control Fault"}

RAW_COLS = [
    "AHU: Cooling Coil Valve Control Signal",
    "AHU: Heating Coil Valve Control Signal",
    "AHU: Mixed Air Temperature",
    "AHU: Outdoor Air Damper Control Signal",
    "AHU: Outdoor Air Temperature",
    "AHU: Return Air Damper Control Signal",
    "AHU: Return Air Temperature",
    "AHU: Supply Air Fan Speed Control Signal",
    "AHU: Supply Air Fan Status",
    "AHU: Supply Air Temperature",
    "Occupancy Mode Indicator",
]

results = {}

def load_and_engineer(path, cap=None):
    df = pd.read_csv(path, nrows=cap, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    label_col = "Fault Detection Ground Truth"
    if label_col not in df.columns:
        return None, None
    cols = [c for c in RAW_COLS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for col in cols:
        for w in [10, 30, 60]:
            df[f"{col}_rm{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_rs{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    if "AHU: Supply Air Temperature" in df.columns and "AHU: Return Air Temperature" in df.columns:
        df["Temp_supply_return_diff"] = df["AHU: Supply Air Temperature"] - df["AHU: Return Air Temperature"]
    if "AHU: Outdoor Air Temperature" in df.columns and "AHU: Supply Air Temperature" in df.columns:
        df["Temp_outdoor_supply_diff"] = df["AHU: Outdoor Air Temperature"] - df["AHU: Supply Air Temperature"]
    if "AHU: Mixed Air Temperature" in df.columns and "AHU: Return Air Temperature" in df.columns:
        df["Temp_mixed_return_diff"] = df["AHU: Mixed Air Temperature"] - df["AHU: Return Air Temperature"]
    feat_cols = [c for c in df.columns if c not in [label_col, "Datetime", "datetime", "Timestamp"]]
    df = df[feat_cols + [label_col]].dropna()
    y = pd.to_numeric(df[label_col], errors="coerce").astype(int)
    return df[feat_cols], y

def compute_psi(ref, dep, n_bins=10):
    ref = pd.to_numeric(ref, errors="coerce").dropna().astype(float)
    dep = pd.to_numeric(dep, errors="coerce").dropna().astype(float)
    if len(ref) == 0 or len(dep) == 0: return 0.0
    mn = float(min(ref.min(), dep.min()))
    mx = float(max(ref.max(), dep.max()))
    if mx == mn: return 0.0
    bins = np.linspace(mn, mx, n_bins+1)
    eps  = 1e-6
    rc   = np.histogram(ref, bins=bins)[0].astype(float) + eps
    dc   = np.histogram(dep, bins=bins)[0].astype(float) + eps
    rp, dp = rc/rc.sum(), dc/dc.sum()
    return float(np.sum((dp-rp)*np.log(dp/rp)))

def report(y_true, y_pred):
    classes = sorted(np.unique(y_true).tolist())
    names   = [FAULT_MAP.get(c, str(c)) for c in classes]
    print(classification_report(y_true, y_pred,
          labels=classes, target_names=names, zero_division=0))

print("=" * 65)
print("THESIS REPRODUCTION SCRIPT")
print("Harshil Patel - bbw Hochschule Berlin - 2026")
print("=" * 65)

# STEP 1: LOAD TRAINING DATA
print("\nSTEP 1: Loading and engineering training data...")
t0 = time.time()
train_X, train_y, train_ref = [], [], []
for f in TRAIN_FILES:
    path = os.path.join(BASE, f)
    if not os.path.exists(path):
        print(f"  [SKIP] {f} not found"); continue
    X, y = load_and_engineer(path)
    if X is not None:
        train_X.append(X)
        train_y.extend(y.tolist())
        train_ref.append(X[[c for c in RAW_COLS if c in X.columns]])
        print(f"  {f}: {len(X):,} rows, {len(X.columns)} features, classes: {sorted(y.unique().tolist())}")

train_df     = pd.concat(train_X,   ignore_index=True)
train_ref_df = pd.concat(train_ref, ignore_index=True)
feat_cols    = list(train_df.columns)
X_train      = train_df.values.astype(float)
y_train      = np.array(train_y, dtype=int)
print(f"\n  Total: {len(X_train):,} rows, {len(feat_cols)} features")
print(f"  Classes in training: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Time: {time.time()-t0:.1f}s")
results["training_rows"]      = int(len(X_train))
results["feature_count"]      = int(len(feat_cols))
results["class_distribution"] = {str(k): int(v) for k,v in zip(*np.unique(y_train, return_counts=True))}

# STEP 2: TRAIN MODEL
print("\nSTEP 2: Training Random Forest (200 trees, balanced weights)...")
t0 = time.time()
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)
print(f"  Training time: {time.time()-t0:.1f}s")

# STEP 3: IN-SAMPLE WITH 20% HOLDOUT
print("\nSTEP 3: In-sample evaluation (20% stratified holdout)...")
X_tr, X_ho, y_tr, y_ho = train_test_split(
    X_train_sc, y_train, test_size=0.2, stratify=y_train, random_state=42)
model_ho = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
model_ho.fit(X_tr, y_tr)
preds_ho    = model_ho.predict(X_ho)
f1_insample = f1_score(y_ho, preds_ho, average="macro", zero_division=0)

print("  Computing bootstrap CI (1000 iterations)...")
rng = np.random.RandomState(42)
boot_f1 = []
for _ in range(1000):
    idx = rng.choice(len(y_ho), len(y_ho), replace=True)
    boot_f1.append(f1_score(y_ho[idx], preds_ho[idx], average="macro", zero_division=0))
ci_lo = float(np.percentile(boot_f1, 2.5))
ci_hi = float(np.percentile(boot_f1, 97.5))

print(f"\n  IN-SAMPLE MACRO-F1 = {f1_insample:.4f}  (thesis: 0.9923)")
print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  (thesis: [0.9913, 0.9933])")
report(y_ho, preds_ho)
results["f1_insample"] = float(f1_insample)
results["ci_lower"]    = ci_lo
results["ci_upper"]    = ci_hi

# STEP 4: CROSS-BUILDING
print("\nSTEP 4: Cross-building evaluation (withheld buildings)...")
test_X, test_y, test_ref = [], [], []
for f in TEST_FILES:
    path = os.path.join(BASE, f)
    if not os.path.exists(path):
        print(f"  [SKIP] {f} not found"); continue
    X, y = load_and_engineer(path, cap=15000)
    if X is not None:
        cols = [c for c in feat_cols if c in X.columns]
        test_X.append(X[cols])
        test_y.extend(y.tolist())
        test_ref.append(X[[c for c in RAW_COLS if c in X.columns]])
        print(f"  {f}: {len(X):,} rows, classes: {sorted(y.unique().tolist())}")

test_df     = pd.concat(test_X,  ignore_index=True)
test_ref_df = pd.concat(test_ref, ignore_index=True)
cols        = [c for c in feat_cols if c in test_df.columns]
X_test_sc   = scaler.transform(test_df[cols].values.astype(float))
y_test      = np.array(test_y, dtype=int)
preds_cross = model.predict(X_test_sc)
f1_cross    = f1_score(y_test, preds_cross, average="macro", zero_division=0)

print(f"\n  CROSS-BUILDING MACRO-F1 = {f1_cross:.4f}  (thesis: 0.331)")
print(f"  Gap = {f1_insample - f1_cross:.4f} pp  (thesis: 66.1 pp)")
report(y_test, preds_cross)
results["f1_crossbuilding"] = float(f1_cross)
results["gap_pp"]           = float(f1_insample - f1_cross)

# STEP 5: PSI ANALYSIS
print("\nSTEP 5: PSI distribution shift analysis (raw sensors)...")
psi_results = {}
print(f"\n  {'Sensor':50}  {'PSI':8}  Category")
print("  " + "-" * 68)
for col in RAW_COLS:
    if col in train_ref_df.columns and col in test_ref_df.columns:
        psi = compute_psi(train_ref_df[col], test_ref_df[col])
        psi_results[col] = psi
        cat = "Extreme" if psi>0.5 else "High" if psi>0.2 else "Moderate" if psi>0.1 else "Low"
        print(f"  {col:50}  {psi:.4f}   {cat}")
results["psi_values"] = {k: round(v,4) for k,v in psi_results.items()}

# STEP 6: SHAP
print("\nSTEP 6: SHAP analysis on full 70-feature model (300 rows, ~3 min)...")
try:
    import shap
    rng2     = np.random.RandomState(42)
    shap_idx = rng2.choice(len(X_test_sc), min(300, len(X_test_sc)), replace=False)
    X_shap   = X_test_sc[shap_idx]
    exp      = shap.TreeExplainer(model)
    sv       = exp.shap_values(X_shap)
    if isinstance(sv, list):
        abs_mean = np.mean([np.abs(s) for s in sv], axis=0)
    elif np.array(sv).ndim == 3:
        abs_mean = np.abs(sv).mean(axis=2)
    else:
        abs_mean = np.abs(sv)
    mean_shap = np.array(abs_mean).mean(axis=0).flatten().astype(float)
    order     = np.argsort(mean_shap)[::-1]
    print(f"\n  {'Rank':4}  {'Feature':52}  SHAP")
    print("  " + "-" * 70)
    shap_out = []
    for rank, fi in enumerate(order[:15], 1):
        fi  = int(fi)
        sv2 = float(mean_shap[fi])
        fn  = feat_cols[fi]
        print(f"  {rank:3}   {fn:52}  {sv2:.4f}")
        shap_out.append({"rank": rank, "feature": fn, "mean_abs_shap": sv2})
    results["shap_values"] = shap_out
except ImportError:
    print("  SHAP not installed. Run: pip install shap")
    results["shap_values"] = "shap not installed"

# SAVE
out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "thesis_reproduction.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("REPRODUCTION COMPLETE")
print("=" * 65)
print(f"  In-sample F1     : {results['f1_insample']:.4f}  (thesis: 0.9923)")
print(f"  95% CI           : [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]  (thesis: [0.9913, 0.9933])")
print(f"  Cross-building F1: {results['f1_crossbuilding']:.4f}  (thesis: 0.331)")
print(f"  Gap              : {results['gap_pp']:.4f} pp  (thesis: 66.1 pp)")
print(f"\n  Full results saved to: {out_path}")
print("=" * 65)
