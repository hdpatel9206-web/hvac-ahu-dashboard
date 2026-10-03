# FIX 4+5 - Cross-building per-class results and full PSI table
# Run: python fix4_crossbuilding_psi.py

import warnings, os, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, classification_report

BASE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVAL_CAP     = 15000
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

# Train model
print("Training model on 3 training buildings...")
train_X, train_y, train_ref = [], [], []
for f in TRAIN_FILES:
    df, y = load(os.path.join(BASE, f))
    if df is not None:
        train_X.append(df)
        train_y.extend(y.tolist())
        train_ref.append(df)
        print(f"  {f}: {len(df):,} rows")

train_df     = pd.concat(train_X,   ignore_index=True)
train_ref_df = pd.concat(train_ref, ignore_index=True)
feat_cols    = list(train_df.columns)
X_train      = train_df.values.astype(float)
y_train      = np.array(train_y, dtype=int)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)
print(f"  In-sample F1 = {f1_score(y_train, model.predict(X_train_sc), average='macro', zero_division=0):.4f}")

# Test on withheld buildings
print("\nEvaluating on withheld buildings...")
test_X_all, test_y_all, test_ref_all = [], [], []
for f in TEST_FILES:
    df, y = load(os.path.join(BASE, f), cap=EVAL_CAP)
    if df is not None:
        cols = [c for c in feat_cols if c in df.columns]
        test_X_all.append(df[cols])
        test_y_all.extend(y.tolist())
        test_ref_all.append(df[cols])
        print(f"  {f}: {len(df):,} rows, classes present: {sorted(y.unique().tolist())}")

test_df      = pd.concat(test_X_all,  ignore_index=True)
test_ref_df  = pd.concat(test_ref_all, ignore_index=True)
cols         = [c for c in feat_cols if c in test_df.columns]
X_test_sc    = scaler.transform(test_df[cols].values.astype(float))
y_test       = np.array(test_y_all, dtype=int)
preds        = model.predict(X_test_sc)

f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

# Get only the classes actually present in the test set
classes_present = sorted(np.unique(y_test).tolist())
class_name_map  = {0:"Healthy", 1:"Sensor Bias", 2:"Valve Fault", 3:"Control Fault"}
labels_present  = classes_present
names_present   = [class_name_map[c] for c in classes_present]

print("\n" + "=" * 65)
print("FIX 4 - CROSS-BUILDING PER-CLASS RESULTS")
print(f"Classes present in withheld test set: {classes_present}")
print(f"Macro-F1 = {f1_macro:.4f}")
print("NOTE: Only classes {classes_present} present — Fault-C absent from test set")
print("=" * 65)
print(classification_report(y_test, preds,
      labels=labels_present,
      target_names=names_present,
      zero_division=0))

# PSI computation
def compute_psi(ref_vals, dep_vals, n_bins=10):
    ref_clean = pd.to_numeric(ref_vals, errors="coerce").dropna().astype(float)
    dep_clean = pd.to_numeric(dep_vals, errors="coerce").dropna().astype(float)
    if len(ref_clean) == 0 or len(dep_clean) == 0:
        return 0.0
    mn = float(min(ref_clean.min(), dep_clean.min()))
    mx = float(max(ref_clean.max(), dep_clean.max()))
    if mx == mn:
        return 0.0
    bins  = np.linspace(mn, mx, n_bins + 1)
    eps   = 1e-6
    ref_c = np.histogram(ref_clean, bins=bins)[0].astype(float) + eps
    dep_c = np.histogram(dep_clean, bins=bins)[0].astype(float) + eps
    ref_p = ref_c / ref_c.sum()
    dep_p = dep_c / dep_c.sum()
    return float(np.sum((dep_p - ref_p) * np.log(dep_p / ref_p)))

print("\n" + "=" * 70)
print("FIX 5 - FULL PSI TABLE FOR ALL RAW SENSORS")
print("Copy into Table 4.5 in Chapter 4")
print("=" * 70)

psi_results = {}
for col in feat_cols:
    if col in train_ref_df.columns and col in test_ref_df.columns:
        psi_results[col] = compute_psi(train_ref_df[col], test_ref_df[col])

psi_sorted = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)

print(f"  {'Sensor Feature':50}  {'PSI':8}  Category")
print("  " + "-" * 70)
for col, psi in psi_sorted:
    cat = "Extreme" if psi > 0.5 else "High" if psi > 0.2 else "Moderate" if psi > 0.1 else "Low"
    print(f"  {col:50}  {psi:.4f}   {cat}")
print("=" * 70)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "crossbuilding_psi_table.json"), "w") as f:
    json.dump({"psi_values": dict(psi_sorted),
               "f1_macro": float(f1_macro),
               "classes_present": classes_present}, f, indent=2)
print(f"\nSaved to results/crossbuilding_psi_table.json")
