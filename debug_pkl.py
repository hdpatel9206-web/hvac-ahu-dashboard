# debug_pkl.py - Find exactly why the pkl fails to load
# Run: python debug_pkl.py

import os, sys
print(f"Python version: {sys.version}")
print(f"Running from: {os.getcwd()}")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
print(f"Model directory: {MODEL_DIR}")

# Check files exist
for fname in ["rf_model.pkl", "scaler.pkl", "feature_cols.pkl"]:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {fname}: EXISTS ({size:,} bytes)")
    else:
        print(f"  {fname}: NOT FOUND at {path}")

print("\nAttempting to load rf_model.pkl...")
try:
    import pickle
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    print(f"  SUCCESS - model type: {type(model)}")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  n_features_in_: {model.n_features_in_}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\nAttempting to load scaler.pkl...")
try:
    import pickle
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print(f"  SUCCESS - scaler type: {type(scaler)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\nAttempting to load feature_cols.pkl...")
try:
    import pickle
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    print(f"  SUCCESS - {len(feature_cols)} features")
    print(f"  First 5: {feature_cols[:5]}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\nChecking scikit-learn version...")
try:
    import sklearn
    print(f"  scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("  scikit-learn NOT installed")

print("\nDone.")
