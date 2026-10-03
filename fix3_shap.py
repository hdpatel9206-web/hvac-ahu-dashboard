# FIX 3 - Real SHAP values from trained model
# Run: python fix3_shap.py
# If shap not installed: pip install shap

import warnings, os, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score

BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVAL_CAP  = 5000
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

# Load training data
print("Step 1: Loading training data...")
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

# Load test data for cross-building evaluation
print("\nStep 2: Loading test data (withheld buildings)...")
test_X, test_y = [], []
for f in TEST_FILES:
    df, y = load(os.path.join(BASE, f), cap=EVAL_CAP)
    if df is not None:
        test_X.append(df)
        test_y.extend(y.tolist())
        print(f"  {f}: {len(df):,} rows")

test_df = pd.concat(test_X, ignore_index=True)
X_test = test_df[feat_cols].values.astype(float)
y_test = np.array(test_y, dtype=int)

print(f"\nStep 3: Training model...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)

f1_train = f1_score(y_train, model.predict(X_train_sc), average="macro", zero_division=0)
f1_cross = f1_score(y_test,  model.predict(X_test_sc),  average="macro", zero_division=0)
print(f"  In-sample F1     = {f1_train:.4f}")
print(f"  Cross-building F1 = {f1_cross:.4f}")
print(f"\n  NOTE: In-sample F1=1.0 is expected here because we have no holdout.")
print(f"  The thesis F1=0.9923 was computed on a 20% stratified holdout sample.")
print(f"  Cross-building F1={f1_cross:.4f} is the key number.")

print("\nStep 4: Computing SHAP values on cross-building test sample (2-5 min)...")
try:
    import shap

    # Use cross-building test sample for SHAP - more representative
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sc)

    # Average absolute SHAP across all classes
    if isinstance(shap_values, list):
        abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_mean = np.abs(shap_values)

    mean_shap = abs_mean.mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]

    print("\n" + "=" * 70)
    print("REAL SHAP VALUES - Copy into Table 4.4 in Chapter 4")
    print("=" * 70)
    print(f"  {'Rank':4} {'Feature':52} {'Mean |SHAP|'}")
    print("  " + "-" * 70)
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:3}   {feat_cols[idx]:52}  {mean_shap[idx]:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    results = [{"rank": int(rank+1),
                "feature": feat_cols[int(sorted_idx[rank])],
                "mean_abs_shap": float(mean_shap[int(sorted_idx[rank])])}
               for rank in range(len(feat_cols))]
    with open(os.path.join(out_dir, "shap_values.json"), "w") as f:
        json.dump({"shap_values": results,
                   "f1_insample": float(f1_train),
                   "f1_crossbuilding": float(f1_cross)}, f, indent=2)
    print(f"\nSaved to results/shap_values.json")

except ImportError:
    print("\nSHAP not installed. Run:")
    print("  pip install shap")
    print("Then run this script again.")
