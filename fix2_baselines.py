# FIX 2 - Baseline Model Comparison for Table 4.2
# Run: python fix2_baselines.py

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.dummy         import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report
from scipy.stats           import binomtest

BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVAL_CAP = 15000
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
        return None
    cols = [c for c in SHARED_COLS if c in df.columns]
    df2 = df[cols + [label_col]].copy()
    for c in cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna()
    df2["y"] = pd.to_numeric(df2[label_col], errors="coerce").astype("Int64")
    df2 = df2.dropna(subset=["y"])
    return df2[cols + ["y"]]

print("Loading training data...")
train_frames = []
for f in TRAIN_FILES:
    df = load(os.path.join(BASE, f))
    if df is not None:
        train_frames.append(df)
        print(f"  {f}: {len(df):,} rows")

train = pd.concat(train_frames, ignore_index=True)
feat_cols = [c for c in SHARED_COLS if c in train.columns]
X_train = train[feat_cols].values.astype(float)
y_train = train["y"].values.astype(int)

print(f"\nClass distribution in training: {dict(zip(*np.unique(y_train, return_counts=True)))}")

print("\nLoading test data (withheld buildings)...")
test_frames = []
for f in TEST_FILES:
    df = load(os.path.join(BASE, f), cap=EVAL_CAP)
    if df is not None:
        test_frames.append(df)
        print(f"  {f}: {len(df):,} rows, classes: {sorted(df['y'].unique().tolist())}")

test = pd.concat(test_frames, ignore_index=True)
X_test = test[feat_cols].values.astype(float)
y_test = test["y"].values.astype(int)

print(f"\nClass distribution in test: {dict(zip(*np.unique(y_test, return_counts=True)))}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\nTraining and evaluating all models on CROSS-BUILDING test set...")
print("=" * 65)
print("NOTE: These F1 values are CROSS-BUILDING (withheld buildings)")
print("      The thesis in-sample F1 = 0.9923 is from a different evaluation")
print("=" * 65)

models = {
    "Random Forest (thesis model)": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "Decision Tree":          DecisionTreeClassifier(
        class_weight="balanced", random_state=42),
    "Logistic Regression":    LogisticRegression(
        class_weight="balanced", max_iter=1000,
        random_state=42, n_jobs=-1),
    "Majority Class Dummy":   DummyClassifier(strategy="most_frequent"),
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    results[name] = f1
    predictions[name] = preds
    print(f"  {name:38s}  Cross-building F1 = {f1:.4f}")

print("=" * 65)
print()
print("NOTE: For TABLE 4.2 in Chapter 4 we need IN-SAMPLE baselines")
print("Now computing in-sample F1 for all models...")
print("=" * 65)

insample = {}
for name, model in models.items():
    preds_train = model.predict(X_train_sc)
    f1 = f1_score(y_train, preds_train, average="macro", zero_division=0)
    insample[name] = f1
    print(f"  {name:38s}  In-sample F1 = {f1:.4f}")

print("=" * 65)

print("\nMcNemar Tests (cross-building) vs Random Forest:")
rf_preds  = predictions["Random Forest (thesis model)"]
rf_correct = (rf_preds == y_test)

for name, preds in predictions.items():
    if name == "Random Forest (thesis model)":
        continue
    other_correct = (preds == y_test)
    b = int(((rf_correct == True)  & (other_correct == False)).sum())
    c = int(((rf_correct == False) & (other_correct == True)).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
        p = result.pvalue
    sig = "SIGNIFICANT p<0.001" if p < 0.001 else f"p={p:.4f}"
    print(f"  RF vs {name:32s}  b={b:5d} c={c:5d}  {sig}")

print("\n--- COPY THESE INTO TABLE 4.2 IN CHAPTER 4 ---")
print("In-sample F1 scores:")
for name, f1 in insample.items():
    print(f"  {name:38s}  {f1:.4f}")

import json
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
    json.dump({"insample": insample, "crossbuilding": results}, f, indent=2)
print(f"\nSaved to results/baseline_results.json")
