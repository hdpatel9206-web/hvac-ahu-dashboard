# FIX 3 FAST VERSION - Real SHAP values
# Run: python pipeline/shap_global_ranking.py

import warnings, os, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

TRAIN_FILES = ["MZVAV-1.csv", "MZVAV-2-1.csv", "SZCAV.csv"]
TEST_FILES  = ["MZVAV-2-2.csv", "SZVAV.csv"]

SHARED_COLS = [
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

def load(path, cap=None):
    df = pd.read_csv(path, nrows=cap, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    label_col = "Fault Detection Ground Truth"
    if label_col not in df.columns:
        return None, None
    cols = [c for c in SHARED_COLS if c in df.columns]
    df2 = df[cols + [label_col]].copy()
    for c in cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna()
    y = pd.to_numeric(df2[label_col], errors="coerce").astype(int)
    return df2[cols], y

print("Loading training data...")
train_X, train_y = [], []
for f in TRAIN_FILES:
    df, y = load(os.path.join(BASE, f))
    if df is not None:
        train_X.append(df)
        train_y.extend(y.tolist())
        print(f"  {f}: {len(df):,} rows")

train_df = pd.concat(train_X, ignore_index=True)
feat_cols = list(train_df.columns)
X_train = train_df.values.astype(float)
y_train = np.array(train_y, dtype=int)

print("\nLoading test data...")
test_X = []
for f in TEST_FILES:
    df, y = load(os.path.join(BASE, f), cap=2000)
    if df is not None:
        test_X.append(df)

test_df = pd.concat(test_X, ignore_index=True)
X_test = test_df[feat_cols].values.astype(float)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Training fast model (50 trees)...")
fast_model = RandomForestClassifier(
    n_estimators=50, class_weight="balanced",
    random_state=42, n_jobs=-1)
fast_model.fit(X_train_sc, y_train)

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_test_sc), min(500, len(X_test_sc)), replace=False)
X_shap = X_test_sc[sample_idx]

print(f"Computing SHAP on {len(X_shap)} rows...")

try:
    import shap

    explainer   = shap.TreeExplainer(fast_model)
    shap_values = explainer.shap_values(X_shap)

    # Handle both old shap (list of arrays) and new shap (3D array)
    if isinstance(shap_values, list):
        # old shap: list of [n_samples, n_features] arrays, one per class
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_shap = stacked.mean(axis=0).mean(axis=0)  # shape: (n_features,)
    elif shap_values.ndim == 3:
        # new shap: [n_samples, n_features, n_classes]
        mean_shap = np.abs(shap_values).mean(axis=2).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    # Ensure 1D float array
    mean_shap = np.array(mean_shap).flatten().astype(float)

    print("\n" + "=" * 68)
    print("REAL SHAP VALUES - Copy into Table 4.4 in Chapter 4")
    print("=" * 68)
    print(f"  Rank  Feature                                             SHAP")
    print("  " + "-" * 68)

    order = np.argsort(mean_shap)[::-1]
    results = []
    for rank, fi in enumerate(order, 1):
        fi = int(fi)
        sv = float(mean_shap[fi])
        print(f"  {rank:3}   {feat_cols[fi]:50}  {sv:.4f}")
        results.append({"rank": rank, "feature": feat_cols[fi], "mean_abs_shap": sv})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "shap_values.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/shap_values.json")

except ImportError:
    print("\nSHAP not installed. Run:  pip install shap")

print("\nDone.")
