"""
RTU Cross-Equipment Generalisation Experiment
Tests whether AHU-trained model generalises to Rooftop Units
RTU = dominant HVAC system in retail stores, supermarkets, small commercial buildings
Harshil Patel - bbw Hochschule Berlin - 2026
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import joblib
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASHRAE_DIR   = os.path.join(REPO_ROOT, 'data')
RTU_FILE     = os.path.join(REPO_ROOT, 'data', 'RTU.csv')
RANDOM_STATE = 42
N_ESTIMATORS = 200
WINDOW_SIZES = [10, 30, 60]
ADAPT_PROPS  = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

TRAIN_FILES = {
    'MZVAV-1.csv' : 1,
    'MZVAV-2-1.csv': 2,
    'SZCAV.csv'   : 3,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean(df):
    df.columns = df.columns.str.strip()
    df = df.replace(['#VALUE!','#REF!','#DIV/0!','#N/A',
                     '#NAME?','#NULL!','#NUM!'], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_shared(df_ashrae, df_rtu):
    """Find sensor columns shared between ASHRAE and RTU."""
    skip = {'Fault Detection Ground Truth','label','Datetime',
            'datetime','Date','date','Time','time'}
    a_cols = set(df_ashrae.columns) - skip
    r_cols = set(df_rtu.columns) - skip
    shared = sorted(a_cols & r_cols)
    return shared

def engineer(df, shared):
    raw = [c for c in shared if c in df.columns]
    data = df[raw].copy().fillna(0)
    frames = [data]
    for w in WINDOW_SIZES:
        rm = data.rolling(w, min_periods=1).mean()
        rs = data.rolling(w, min_periods=1).std().fillna(0)
        rm.columns = [f'{c}_rm{w}' for c in raw]
        rs.columns = [f'{c}_rs{w}' for c in raw]
        frames += [rm, rs]
    feat = pd.concat(frames, axis=1)

    # Derived temperature features
    sa = next((c for c in raw if 'Supply Air Temp' in c), None)
    ra = next((c for c in raw if 'Return Air Temp' in c), None)
    oa = next((c for c in raw if 'Outdoor Air Temp' in c), None)
    ma = next((c for c in raw if 'Mixed Air Temp' in c), None)

    if sa and ra:
        feat['diff_sa_ra'] = df[sa].values - df[ra].values
    if oa and sa:
        feat['diff_oa_sa'] = df[oa].values - df[sa].values
    if ma and ra:
        feat['diff_ma_ra'] = df[ma].values - df[ra].values

    return feat.fillna(0)

def compute_psi(ref, tgt, n_bins=10, eps=1e-6):
    lo = min(ref.min(), tgt.min())
    hi = max(ref.max(), tgt.max())
    if lo == hi: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    rp = np.histogram(ref, bins=bins)[0].astype(float) + eps
    tp = np.histogram(tgt, bins=bins)[0].astype(float) + eps
    rp /= rp.sum(); tp /= tp.sum()
    return float(np.sum((tp - rp) * np.log(tp / rp)))

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_ashrae(shared):
    frames = []
    for fname, fault_cls in TRAIN_FILES.items():
        fpath = os.path.join(ASHRAE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df = clean(df)
        label = np.where(
            df['Fault Detection Ground Truth'] == 0, 0, fault_cls
        ) if 'Fault Detection Ground Truth' in df.columns \
          else np.zeros(len(df), int)
        feat = engineer(df, shared)
        feat['label'] = label
        frames.append(feat)
        print(f"  {fname}: {len(feat):,} rows "
              f"(H={(label==0).sum():,} F={(label>0).sum():,})")
    return pd.concat(frames, ignore_index=True)

def load_rtu(shared):
    df = pd.read_csv(RTU_FILE, low_memory=False)
    df = clean(df)
    print(f"  RTU.csv: {len(df):,} rows")
    print(f"  RTU columns: {list(df.columns)}")

    # Assign label from ground truth if available
    if 'Fault Detection Ground Truth' in df.columns:
        label = df['Fault Detection Ground Truth'].fillna(0).astype(int)
        # Map: 0=healthy, anything else=fault (class 1 for RTU)
        label = np.where(label == 0, 0, 1)
        print(f"  Using ground truth labels: H={(label==0).sum():,} F={(label>0).sum():,}")
    else:
        # No labels — use as OOD test only
        label = np.zeros(len(df), int)
        print("  No ground truth labels found — using as OOD confidence test")

    feat = engineer(df, shared)
    feat['label'] = label
    return feat, 'labelled' if 'Fault Detection Ground Truth' in df.columns else 'unlabelled'

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("RTU CROSS-EQUIPMENT GENERALISATION EXPERIMENT")
    print("AHU-trained model tested on Rooftop Unit (retail/commercial)")
    print("=" * 65)

    # Load RTU to see its columns
    print("\nStep 1 — Inspecting RTU.csv columns...")
    rtu_raw = pd.read_csv(RTU_FILE, low_memory=False)
    rtu_raw = clean(rtu_raw)
    print(f"  RTU columns ({len(rtu_raw.columns)}):")
    for c in rtu_raw.columns:
        print(f"    {c}")

    # Load one ASHRAE file to find shared columns
    ashrae_sample = clean(pd.read_csv(
        os.path.join(ASHRAE_DIR, 'MZVAV-1.csv'), low_memory=False))

    shared = get_shared(ashrae_sample, rtu_raw)
    print(f"\nShared sensor columns ({len(shared)}):")
    for c in shared:
        print(f"  ✓ {c}")

    if len(shared) == 0:
        print("\n  WARNING: No shared columns found.")
        print("  RTU uses different column names than ASHRAE AHU.")
        print("  Running OOD confidence test instead.")
        run_ood_test(rtu_raw)
        return

    # Load all data
    print(f"\nStep 2 — Loading ASHRAE training data...")
    ashrae = load_ashrae(shared)
    print(f"\nStep 3 — Loading RTU data...")
    rtu, rtu_mode = load_rtu(shared)

    feat_cols = [c for c in ashrae.columns if c != 'label']
    print(f"\nFeatures: {len(feat_cols)}")

    # Train base model on ASHRAE
    print(f"\nStep 4 — Training AHU base model...")
    y_tr = ashrae['label'].values
    X_tr = ashrae[feat_cols]

    rng = np.random.default_rng(RANDOM_STATE)
    sub_X, sub_y = [], []
    for cls in np.unique(y_tr):
        idx = np.where(y_tr == cls)[0]
        n   = min(len(idx), 30000)
        ch  = rng.choice(idx, n, replace=False)
        sub_X.append(X_tr.iloc[ch])
        sub_y.append(y_tr[ch])
    X_base = pd.concat(sub_X).values
    y_base = np.concatenate(sub_y)

    scaler    = StandardScaler()
    X_base_sc = scaler.fit_transform(X_base)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_base_sc, y_base)

    f1_is = f1_score(clf.predict(X_base_sc), y_base,
                     average='macro', zero_division=0)
    print(f"  In-sample F1: {f1_is:.4f}")

    # RTU evaluation
    print(f"\nStep 5 — Cross-equipment evaluation on RTU...")
    y_rtu = rtu['label'].values
    X_rtu = rtu[feat_cols].values
    X_rtu_sc = scaler.transform(X_rtu)

    # PSI analysis
    print(f"\nStep 6 — PSI analysis (AHU training vs RTU)...")
    psi_scores = {}
    for i, col in enumerate(feat_cols):
        psi_scores[col] = compute_psi(X_base_sc[:, i], X_rtu_sc[:, i])

    psi_s = pd.Series(psi_scores).sort_values(ascending=False)
    print("\n  Top 10 PSI features (AHU → RTU shift):")
    for fn, psi in psi_s.head(10).items():
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn[:50]:<50} PSI={psi:.3f}  {tag}")

    raw_psi = {k: v for k, v in psi_scores.items()
               if '_rm' not in k and '_rs' not in k
               and 'diff' not in k}
    print("\n  Raw sensor PSI:")
    for fn, psi in sorted(raw_psi.items(), key=lambda x: -x[1]):
        tag = "EXTREME" if psi > 0.5 else "HIGH" if psi > 0.2 else "LOW"
        print(f"    {fn:<50} PSI={psi:.3f}  {tag}")

    if rtu_mode == 'labelled':
        # Full classification evaluation
        np.random.seed(RANDOM_STATE)
        n_eval = min(len(y_rtu), 15000)
        idx    = np.random.choice(len(y_rtu), n_eval, replace=False)
        yp     = clf.predict(X_rtu_sc[idx])
        f1     = f1_score(y_rtu[idx], yp, average='macro', zero_division=0)

        print(f"\n  Cross-equipment F1 (AHU model on RTU): {f1:.4f}")
        print(classification_report(y_rtu[idx], yp, zero_division=0))

        # Adaptation loop
        print(f"\nStep 7 — Adaptation experiment...")
        eval_idx  = idx
        adapt_pool = np.array([i for i in range(len(y_rtu)) if i not in set(idx)])

        X_eval_sc = X_rtu_sc[eval_idx]
        y_eval    = y_rtu[eval_idx]

        print(f"\n  {'%':>6}  {'Records':>10}  {'F1':>8}  {'Δ':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")
        print(f"  {'0%':>6}  {'0':>10}  {f1:>8.4f}  {'—':>8}")

        curve = [{'proportion':0.0,'records':0,
                  'f1_macro':round(float(f1),4),'delta':0.0}]

        for prop in ADAPT_PROPS[1:]:
            n_adapt = int(len(adapt_pool) * prop)
            if n_adapt == 0: continue
            a_idx   = rng.choice(adapt_pool,
                                 min(n_adapt, len(adapt_pool)), replace=False)
            X_a_sc  = scaler.transform(X_rtu[a_idx])
            y_a     = y_rtu[a_idx]

            X_c = np.vstack([X_base_sc, X_a_sc])
            y_c = np.concatenate([y_base, y_a])

            clf_a = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=-1)
            clf_a.fit(X_c, y_c)

            yp_a  = clf_a.predict(X_eval_sc)
            f1_a  = f1_score(y_eval, yp_a, average='macro', zero_division=0)
            delta = f1_a - f1

            print(f"  {prop:>6.0%}  {n_adapt:>10,}  {f1_a:>8.4f}  {delta:>+8.4f}")
            curve.append({'proportion':prop,'records':n_adapt,
                         'f1_macro':round(float(f1_a),4),
                         'delta':round(float(delta),4)})

        best = max(curve, key=lambda x: x['f1_macro'])

    else:
        # OOD confidence test
        probs = clf.predict_proba(X_rtu_sc)
        max_conf = probs.max(axis=1)
        pred_cls = clf.predict(X_rtu_sc)
        f1 = None
        print(f"\n  OOD Confidence Analysis (no ground truth labels):")
        print(f"  Mean max confidence:  {max_conf.mean():.4f}")
        print(f"  Median confidence:    {np.median(max_conf):.4f}")
        print(f"  Low confidence (<0.5): {(max_conf<0.5).mean()*100:.1f}%")
        print(f"  Predicted class distribution:")
        for cls, cnt in zip(*np.unique(pred_cls, return_counts=True)):
            print(f"    Class {cls}: {cnt:,} ({cnt/len(pred_cls)*100:.1f}%)")
        curve = []
        best = {}

    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Equipment type:              AHU (train) → RTU (test)")
    print(f"  In-sample F1:               {f1_is:.4f}")
    print(f"  Thesis AHU→AHU CB F1:       0.331")
    if f1 is not None:
        print(f"  AHU→RTU cross-equip F1:    {f1:.4f}")
        print(f"  Best after adaptation:      {best.get('f1_macro','—')}")
    print(f"  Shared features:            {len(feat_cols)}")
    print()
    if f1 is not None:
        if f1 > 0.40:
            print("  FINDING: Better than AHU cross-building baseline (0.331)")
            print("  RTU fault signatures partially share structure with AHU faults")
        elif f1 < 0.20:
            print("  FINDING: Worse than AHU cross-building — equipment gap is larger")
            print("  RTU control logic and sensor architecture are fundamentally different")
        else:
            print("  FINDING: Similar to AHU cross-building baseline (~0.33)")
            print("  Distribution shift magnitude is consistent across equipment types")

    results = {
        'experiment'        : 'rtu_cross_equipment',
        'train_equipment'   : 'AHU (ASHRAE MZVAV + SZCAV)',
        'test_equipment'    : 'RTU (retail/commercial)',
        'f1_insample'       : round(float(f1_is), 4),
        'f1_cross_equip'    : round(float(f1), 4) if f1 else None,
        'thesis_ahu_cb_f1'  : 0.331,
        'shared_features'   : len(feat_cols),
        'top5_psi'          : psi_s.head(5).round(3).to_dict(),
        'adaptation_curve'  : curve,
        'rtu_mode'          : rtu_mode,
    }
    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    with open(os.path.join(REPO_ROOT, 'results', 'rtu_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/rtu_experiment.json")
    print("=" * 65)

def run_ood_test(rtu_raw):
    """Fallback: load thesis model and run OOD confidence test on RTU."""
    print("\nRunning OOD confidence test with thesis model...")
    model_path  = os.path.join(ASHRAE_DIR, '..', 'models', 'rf_model.pkl')
    scaler_path = os.path.join(ASHRAE_DIR, '..', 'models', 'scaler.pkl')
    feat_path   = os.path.join(ASHRAE_DIR, '..', 'models', 'feature_cols.pkl')

    if not os.path.exists(model_path):
        print("  Model not found. Check ASHRAE_DIR path.")
        return

    model     = joblib.load(model_path)
    scaler    = joblib.load(scaler_path)
    feat_cols = joblib.load(feat_path)

    # Zero-fill missing features
    X = pd.DataFrame(0.0, index=rtu_raw.index, columns=feat_cols)
    for c in feat_cols:
        if c in rtu_raw.columns:
            X[c] = rtu_raw[c].values

    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)
    preds = model.predict(X_sc)
    max_conf = probs.max(axis=1)

    print(f"  Mean confidence:       {max_conf.mean():.4f}")
    print(f"  Median confidence:     {np.median(max_conf):.4f}")
    print(f"  Low confidence (<0.5): {(max_conf<0.5).mean()*100:.1f}%")
    print(f"  Class distribution:")
    for cls, cnt in zip(*np.unique(preds, return_counts=True)):
        names = {0:'Healthy',1:'Fault-A',2:'Fault-B',3:'Fault-C'}
        print(f"    {names.get(cls,'Class '+str(cls))}: {cnt:,} ({cnt/len(preds)*100:.1f}%)")

if __name__ == '__main__':
    main()
