"""
precompute_shap.py
==================
Computes SHAP values ONCE and writes them to results/shap_dashboard.json.
The dashboard then loads the JSON instead of recomputing, which turns a
15-minute page load into an instant one.

Also produces the stored artefact for Table 4.4 traceability.

Run:  python precompute_shap.py
"""
import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

BASE = r"C:\Users\hrslp\Desktop\thesis"
MODEL = os.path.join(BASE, "models", "rf_model.pkl")
FEATS = os.path.join(BASE, "models", "feature_cols.pkl")
SCALER = os.path.join(BASE, "models", "scaler.pkl")
OUT = os.path.join(BASE, "results", "shap_dashboard.json")

# keep this small — SHAP cost is linear in sample count
N_SAMPLES = 200

CLASS_NAMES = ["Healthy", "Fault-A (OA Sensor)", "Fault-B (Valve Leak)", "Fault-C (Control)"]


def build_feature_matrix(feat_cols):
    """
    Reconstruct the engineered feature matrix for the cross-building test set.
    Mirrors the pipeline: raw channels -> rolling stats -> derived features.
    """
    RAW = [
        'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
        'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
        'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
        'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
        'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
        'Occupancy Mode Indicator',
    ]
    WINDOWS = [10, 30, 60]
    NO_ROLL = {'AHU: Outdoor Air Damper Control Signal',
               'AHU: Return Air Damper Control Signal'}

    frames = []
    for fname in ("MZVAV-2-2.csv", "SZVAV.csv"):
        path = os.path.join(BASE, "data", fname)
        if not os.path.exists(path):
            print(f"  ! missing {fname}, skipping")
            continue
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        avail = [c for c in RAW if c in df.columns]
        out = df[avail].apply(pd.to_numeric, errors="coerce").fillna(0)

        for col in avail:
            if col in NO_ROLL:
                continue
            for w in WINDOWS:
                out[f"{col}_rm{w}"] = out[col].rolling(w, min_periods=1).mean()
                out[f"{col}_rs{w}"] = out[col].rolling(w, min_periods=1).std().fillna(0)

        def diff(a, b):
            return out[a] - out[b] if a in out and b in out else 0.0

        out["Temp_supply_return_diff"]  = diff('AHU: Supply Air Temperature', 'AHU: Return Air Temperature')
        out["Temp_outdoor_supply_diff"] = diff('AHU: Outdoor Air Temperature', 'AHU: Supply Air Temperature')
        out["Temp_mixed_return_diff"]   = diff('AHU: Mixed Air Temperature', 'AHU: Return Air Temperature')
        out["Valve_diff"]               = diff('AHU: Cooling Coil Valve Control Signal',
                                               'AHU: Heating Coil Valve Control Signal')
        fan = 'AHU: Supply Air Fan Speed Control Signal'
        dmp = 'AHU: Outdoor Air Damper Control Signal'
        out["Fan_damper_ratio"] = (out[fan] / out[dmp].replace(0, np.nan)).fillna(0) \
                                  if fan in out and dmp in out else 0.0
        frames.append(out)

    if not frames:
        raise SystemExit("No data files found — check the data/ path.")

    X = pd.concat(frames, ignore_index=True)
    for c in feat_cols:                       # add anything missing as zeros
        if c not in X.columns:
            X[c] = 0.0
    return X[feat_cols]                       # exact training column order


def main():
    print("Loading model artefacts ...")
    model = joblib.load(MODEL)
    feat_cols = joblib.load(FEATS)
    print(f"  {len(model.estimators_)} trees, {len(feat_cols)} features")

    depths = [t.tree_.max_depth for t in model.estimators_]
    leaves = [t.tree_.n_leaves for t in model.estimators_]
    print(f"  depth  mean {np.mean(depths):.1f}  max {max(depths)}")
    print(f"  leaves mean {np.mean(leaves):.0f}  max {max(leaves)}")
    print("  (deep trees are why TreeSHAP is slow — hence precomputing)")

    print("\nBuilding feature matrix ...")
    X = build_feature_matrix(feat_cols)
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(N_SAMPLES, len(X)), replace=False)
    Xs = X.iloc[idx]

    if os.path.exists(SCALER):
        Xs = pd.DataFrame(joblib.load(SCALER).transform(Xs), columns=feat_cols)
        print("  scaler applied")

    print(f"\nComputing SHAP on {len(Xs)} samples ...")
    import shap
    t0 = time.time()
    # No background data -> tree_path_dependent, which is the fast path.
    # check_additivity=False skips a second full pass.
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs, check_additivity=False)
    print(f"  done in {time.time() - t0:.1f}s")

    # normalise shape across shap versions -> (n_classes, n_samples, n_features)
    if isinstance(sv, list):
        arr = np.array(sv)
    else:
        arr = np.array(sv)
        if arr.ndim == 3 and arr.shape[-1] == len(getattr(model, "classes_", [0])):
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:
            arr = arr[None, ...]

    n_cls = arr.shape[0]
    global_imp = np.abs(arr).mean(axis=(0, 1))          # mean |SHAP| per feature
    order = np.argsort(global_imp)[::-1][:15]

    per_class = {}
    for c in range(n_cls):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        m = np.abs(arr[c]).mean(axis=0)
        per_class[name] = {feat_cols[i]: round(float(m[i]), 6) for i in order}

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_model": "models/rf_model.pkl",
        "n_samples": int(len(Xs)),
        "n_features": int(len(feat_cols)),
        "n_classes": int(n_cls),
        "method": "shap.TreeExplainer, tree_path_dependent",
        "seed": 42,
        "global_top15": {feat_cols[i]: round(float(global_imp[i]), 6) for i in order},
        "per_class_top15": per_class,
        "heatmap_features": [feat_cols[i] for i in order[:10]],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {OUT}")
    print("\nTop 12 features by mean |SHAP|:")
    for i in order[:12]:
        print(f"  {feat_cols[i]:<48} {global_imp[i]:.4f}")


if __name__ == "__main__":
    main()
