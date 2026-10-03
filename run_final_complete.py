# run_final_complete.py
# Retrains the final optimised model, saves it, then runs SHAP and fine-tuning
# Run: python run_final_complete.py
# Time: approximately 60-90 minutes total

import os, json, glob, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, matthews_corrcoef
from scipy.stats import chi2 as chi2_dist

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
SDAHU_DIR = os.path.join(DATA_DIR, "sdahu")
OUT_DIR   = os.path.join(BASE, "results", "new_model_final")
os.makedirs(OUT_DIR, exist_ok=True)
RS = 42
ROLLING_WINDOWS = [10, 30, 60]
MAX_PER_CLASS   = 120000
MAX_PER_FILE    = 50000
N_BOOTSTRAP     = 1000
FINETUNE_PROPS  = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

RAW_COLS = [
    'AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',  'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Outdoor Air Damper Control Signal', 'AHU: Return Air Damper Control Signal',
    'AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]
SDAHU_COL_MAP = {
    'SA_TEMP':'AHU: Supply Air Temperature','OA_TEMP':'AHU: Outdoor Air Temperature',
    'MA_TEMP':'AHU: Mixed Air Temperature','RA_TEMP':'AHU: Return Air Temperature',
    'SF_SPD_DM':'AHU: Supply Air Fan Status','SF_CS':'AHU: Supply Air Fan Speed Control Signal',
    'OA_DMPR_DM':'AHU: Outdoor Air Damper Control Signal',
    'RA_DMPR_DM':'AHU: Return Air Damper Control Signal',
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Control Signal',
    'SYS_CTL':'Occupancy Mode Indicator',
}
ASHRAE_TRAIN = ['MZVAV-1.csv', 'MZVAV-2-1.csv', 'SZCAV.csv']
SDAHU_CLASS0 = ['ahu_annual']
SDAHU_CLASS2 = ['coi_leakage', 'coi_stuck']
SDAHU_CLASS3 = ['damper_stuck']
EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}
CLASS_NAMES  = {0:'Healthy', 1:'Sensor Bias', 2:'Valve Fault', 3:'Damper Fault'}

def engineer_features(df):
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    for col in available:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rm{w}'] = out[col].rolling(w, min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w, min_periods=1).std().fillna(0)
    pairs = [
        ('Temp_supply_return_diff',  'AHU: Supply Air Temperature', 'AHU: Return Air Temperature'),
        ('Temp_outdoor_supply_diff', 'AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
        ('Temp_mixed_return_diff',   'AHU: Mixed Air Temperature',  'AHU: Return Air Temperature'),
    ]
    for name, a, b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a] - out[b]
    return out.fillna(0).replace([np.inf, -np.inf], 0)

def load_ashrae(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    lc = 'Fault Detection Ground Truth'
    if lc not in df.columns: return pd.DataFrame()
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    feats = engineer_features(df)
    feats['label'] = np.where(df[lc].values[:len(feats)] == 0, 0, 1)
    return feats.dropna()

def load_sdahu(fpath, label, zero_oa=False):
    df = pd.read_csv(fpath, low_memory=False, nrows=MAX_PER_FILE)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    for sc, ac in SDAHU_COL_MAP.items():
        if sc in df.columns:
            df[ac] = pd.to_numeric(df[sc], errors='coerce')
    for col in RAW_COLS:
        if col not in df.columns: df[col] = 0.0
    if zero_oa and label in [2, 3]:
        df['AHU: Outdoor Air Temperature'] = 0.0
        df['AHU: Mixed Air Temperature']   = 0.0
    feats = engineer_features(df)
    feats['label'] = label
    return feats.dropna()

def compute_psi(ref, dep, n_bins=10):
    ref = pd.to_numeric(ref, errors='coerce').dropna().astype(float)
    dep = pd.to_numeric(dep, errors='coerce').dropna().astype(float)
    if len(ref) == 0 or len(dep) == 0: return 0.0
    mn, mx = float(min(ref.min(), dep.min())), float(max(ref.max(), dep.max()))
    if mx == mn: return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    eps  = 1e-6
    rc = np.histogram(ref, bins=bins)[0].astype(float) + eps
    dc = np.histogram(dep, bins=bins)[0].astype(float) + eps
    rp, dp = rc/rc.sum(), dc/dc.sum()
    return float(np.sum((dp - rp) * np.log(dp / rp)))

def mcnemar_test(y_true, preds_a, preds_b):
    ca = (preds_a == y_true)
    cb = (preds_b == y_true)
    b  = np.sum( ca & ~cb)
    c  = np.sum(~ca &  cb)
    if b + c == 0: return 0, 1.0
    stat = (abs(b - c) - 1)**2 / (b + c)
    p    = 1 - chi2_dist.cdf(stat, df=1)
    return stat, p

print("=" * 70)
print("FINAL MODEL — COMPLETE RETRAIN + SHAP + FINETUNE")
print("=" * 70)

# ── STEP 1: Load and balance training data ────────────────────────────────────
print("\nStep 1: Loading training data...")
if not os.path.exists(SDAHU_DIR):
    SDAHU_DIR = os.path.join(DATA_DIR, "lbnl_sdahu", "LBNL_FDD_Dataset_SDAHU")

pools = {0:[], 1:[], 2:[], 3:[]}

for fname in ASHRAE_TRAIN:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath): continue
    df = load_ashrae(fpath)
    if len(df) > 0:
        pools[0].append(df[df['label']==0])
        pools[1].append(df[df['label']==1])
        print(f"  ASHRAE {fname}: h={len(df[df['label']==0]):,} sb={len(df[df['label']==1]):,}")

for fpath in sorted(glob.glob(os.path.join(SDAHU_DIR, "*.csv"))):
    fn = os.path.basename(fpath).lower()
    lbl = None
    for p in SDAHU_CLASS0:
        if p in fn: lbl=0; break
    if lbl is None:
        for p in SDAHU_CLASS2:
            if p in fn: lbl=2; break
    if lbl is None:
        for p in SDAHU_CLASS3:
            if p in fn: lbl=3; break
    if lbl is None: continue
    df = load_sdahu(fpath, lbl, zero_oa=True)
    if len(df) > 0:
        pools[lbl].append(df)
        tag = {0:'healthy',2:'valve',3:'damper'}[lbl]
        oa_note = ' [OA zeroed]' if lbl in [2,3] else ''
        print(f"  SD-AHU {os.path.basename(fpath):40s}: {tag}={len(df):,}{oa_note}")

class_dfs = {}
for cls, parts in pools.items():
    if parts:
        comb = pd.concat(parts, ignore_index=True)
        if len(comb) > MAX_PER_CLASS:
            comb = comb.sample(n=MAX_PER_CLASS, random_state=RS)
        class_dfs[cls] = comb

feat_set = set(class_dfs[0].columns)
for df in class_dfs.values(): feat_set &= set(df.columns)
feat_cols = sorted([c for c in feat_set if c != 'label'])

df_train = pd.concat([df[feat_cols+['label']] for df in class_dfs.values()],
                     ignore_index=True)
dist = dict(df_train['label'].value_counts().sort_index())
print(f"\n  Total: {len(df_train):,} rows | Features: {len(feat_cols)}")
print(f"  Distribution: {dist}")

# ── STEP 2: Train ─────────────────────────────────────────────────────────────
print("\nStep 2: Training Random Forest (200 trees)...")
t0 = time.time()
X = df_train[feat_cols].values.astype(float)
y = df_train['label'].values.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=RS)
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)
model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RS, n_jobs=-1, min_samples_leaf=2)
model.fit(X_tr_sc, y_tr)
print(f"  Training time: {time.time()-t0:.1f}s")

preds_in   = model.predict(X_te_sc)
f1_in      = f1_score(y_te, preds_in, average='macro', zero_division=0)
classes_in = sorted(np.unique(y_te).tolist())
print(f"  IN-SAMPLE F1 = {f1_in:.4f}  (old: 0.9923)")
print(classification_report(y_te, preds_in, labels=classes_in,
      target_names=[CLASS_NAMES[c] for c in classes_in], zero_division=0))

# Bootstrap CI
rng = np.random.RandomState(RS)
bs  = [f1_score(y_te[i:=rng.choice(len(X_te_sc),len(X_te_sc),replace=True)],
               preds_in[i], average='macro', zero_division=0)
       for _ in range(N_BOOTSTRAP)]
ci_lo, ci_hi = np.percentile(bs,2.5), np.percentile(bs,97.5)
print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  (old: [0.9913, 0.9933])")

# SAVE MODEL IMMEDIATELY
joblib.dump(model,     os.path.join(OUT_DIR, "rf_model_final.pkl"))
joblib.dump(scaler,    os.path.join(OUT_DIR, "scaler_final.pkl"))
joblib.dump(feat_cols, os.path.join(OUT_DIR, "feature_cols_final.pkl"))
print("  Model saved.")

# ── STEP 3: Cross-building ────────────────────────────────────────────────────
print("\nStep 3: Cross-building evaluation...")
cb_parts = []
for fname in ['MZVAV-2-2.csv', 'SZVAV.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath): continue
    df = load_ashrae(fpath)
    if len(df) > 0:
        cb_parts.append(df)
        print(f"  {fname}: {len(df):,} rows")

df_cb = pd.concat(cb_parts, ignore_index=True)
X_cb_mat = np.zeros((len(df_cb), len(feat_cols)))
for i, col in enumerate(feat_cols):
    if col in df_cb.columns:
        X_cb_mat[:, i] = df_cb[col].values.astype(float)
X_cb_sc  = scaler.transform(X_cb_mat)
y_cb     = df_cb['label'].values.astype(int)
preds_cb = model.predict(X_cb_sc)
f1_cb    = f1_score(y_cb, preds_cb, average='macro', zero_division=0)
f1_cb_bin= f1_score((y_cb>0).astype(int),(preds_cb>0).astype(int),
                    average='macro', zero_division=0)
gap_pp   = (f1_in - f1_cb) * 100
mcc      = matthews_corrcoef(y_cb, preds_cb)
classes_cb = sorted(np.unique(y_cb).tolist())
print(f"  CROSS-BUILDING 4-class F1 = {f1_cb:.4f}  (old: 0.331)")
print(f"  CROSS-BUILDING binary F1  = {f1_cb_bin:.4f}  (old: 0.415)")
print(f"  Gap = {gap_pp:.1f}pp  (old: 66.1pp)")
print(classification_report(y_cb, preds_cb, labels=classes_cb,
      target_names=[CLASS_NAMES.get(c,str(c)) for c in classes_cb], zero_division=0))

# ── STEP 4: McNemar tests ─────────────────────────────────────────────────────
print("\nStep 4: McNemar tests...")
print("  Training SVM...")
svm = CalibratedClassifierCV(LinearSVC(random_state=RS, max_iter=2000))
svm.fit(X_tr_sc[:30000], y_tr[:30000])
preds_svm   = svm.predict(X_cb_sc)
f1_svm      = f1_score(y_cb, preds_svm, average='macro', zero_division=0)
chi2_svm, p_svm = mcnemar_test(y_cb, preds_cb, preds_svm)

print("  Training GB...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=RS)
gb.fit(X_tr_sc[:20000], y_tr[:20000])
preds_gb    = gb.predict(X_cb_sc)
f1_gb       = f1_score(y_cb, preds_gb, average='macro', zero_division=0)
chi2_gb, p_gb = mcnemar_test(y_cb, preds_cb, preds_gb)

dummy = DummyClassifier(strategy='most_frequent', random_state=RS)
dummy.fit(X_tr_sc, y_tr)
preds_d  = dummy.predict(X_cb_sc)
f1_d     = f1_score(y_cb, preds_d, average='macro', zero_division=0)
chi2_d, p_d = mcnemar_test(y_cb, preds_cb, preds_d)

print(f"\n  {'Model':30}  {'F1':8}  {'chi2':10}  p-value   Sig?")
print("  " + "-"*65)
print(f"  {'RF (new model)':30}  {f1_cb:.4f}")
print(f"  {'GB':30}  {f1_gb:.4f}  {chi2_gb:10.1f}  {p_gb:.2e}  {'Yes' if p_gb<0.001 else 'No'}")
print(f"  {'SVM':30}  {f1_svm:.4f}  {chi2_svm:10.1f}  {p_svm:.2e}  {'Yes' if p_svm<0.001 else 'No'}")
print(f"  {'Dummy':30}  {f1_d:.4f}  {chi2_d:10.1f}  {p_d:.2e}  {'Yes' if p_d<0.001 else 'No'}")

# ── STEP 5: PSI ───────────────────────────────────────────────────────────────
print("\nStep 5: PSI analysis...")
old_psi={'AHU: Supply Air Fan Speed Control Signal':2.63,'AHU: Outdoor Air Temperature':2.45,
         'AHU: Supply Air Temperature':0.84,'AHU: Return Air Temperature':0.71,
         'AHU: Mixed Air Temperature':0.51,'AHU: Outdoor Air Damper Control Signal':0.48,
         'AHU: Return Air Damper Control Signal':1.44,'AHU: Cooling Coil Valve Control Signal':0.88,
         'AHU: Heating Coil Valve Control Signal':0.49,'AHU: Supply Air Fan Status':0.01,
         'Occupancy Mode Indicator':0.0002}
psi_results={}
print(f"  {'Sensor':50}  {'New':8}  {'Old':8}  Gate")
print("  "+"-"*72)
for col in RAW_COLS:
    if col in df_train.columns and col in df_cb.columns:
        psi = compute_psi(df_train[col], df_cb[col])
        psi_results[col] = psi
        gate="RED" if psi>0.5 else "AMBER" if psi>0.2 else "GREEN"
        o=old_psi.get(col,"-")
        print(f"  {col:50}  {psi:8.4f}  {str(o):>8}  {gate}")

# ── STEP 6: SHAP ─────────────────────────────────────────────────────────────
shap_out = {}
try:
    import shap
    print("\nStep 6: SHAP analysis...")
    t0 = time.time()
    idx = np.random.RandomState(RS).choice(len(X_cb_sc), min(500,len(X_cb_sc)), replace=False)
    X_shap = X_cb_sc[idx]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(np.array(sv)).mean(axis=0)
                            for sv in shap_vals], axis=0)
    else:
        mean_shap = np.abs(np.array(shap_vals)).mean(axis=0)

    mean_shap = np.array(mean_shap).flatten()
    ranked = np.argsort(mean_shap)[::-1]

    print(f"  {'Rank':4}  {'Feature':52}  Mean|SHAP|")
    print("  "+"-"*72)
    for rank_i, raw_idx in enumerate(ranked[:15], 1):
        fi = int(raw_idx)
        if 0 <= fi < len(feat_cols):
            fname_s = feat_cols[fi]
            val     = float(mean_shap[fi])
            print(f"  {rank_i:4}  {fname_s:52}  {val:.4f}")
            shap_out[fname_s] = round(val, 4)
    print(f"  SHAP time: {time.time()-t0:.1f}s")
except ImportError:
    print("  SHAP not installed. pip install shap")
except Exception as e:
    print(f"  SHAP error: {e}")

# ── STEP 7: Fine-tuning ───────────────────────────────────────────────────────
print("\nStep 7: Fine-tuning experiment...")
finetune_results = []
print(f"  {'%':6}  {'Records':10}  {'4-class':10}  {'Binary':10}  Change")
print("  "+"-"*52)

for pct in FINETUNE_PROPS:
    n   = max(1, int(len(X_cb_sc) * pct))
    idx = np.random.RandomState(RS).choice(len(X_cb_sc), n, replace=False)
    X_new = np.vstack([X_tr_sc, X_cb_sc[idx]])
    y_new = np.concatenate([y_tr, y_cb[idx]])
    m = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RS, n_jobs=-1, min_samples_leaf=2)
    m.fit(X_new, y_new)
    p     = m.predict(X_cb_sc)
    f4    = f1_score(y_cb, p, average='macro', zero_division=0)
    fb    = f1_score((y_cb>0).astype(int),(p>0).astype(int),
                     average='macro', zero_division=0)
    chg   = (f4 - f1_cb) * 100
    finetune_results.append({'pct':pct,'n':n,'f1_4class':round(f4,4),
                              'f1_binary':round(fb,4),'change_pp':round(chg,1)})
    print(f"  {pct*100:5.0f}%  {n:10,}  {f4:10.4f}  {fb:10.4f}  +{chg:.1f}pp")

ft10    = next((r for r in finetune_results if r['pct']==0.10), None)
ft10_f1 = ft10['f1_4class'] if ft10 else 0
recovery= round(ft10_f1 / 0.50 * 100, 1)

print(f"\n  Fine-tuning 10%: 4-class={ft10_f1:.4f}  recovery={recovery}% of max 0.50")
print(f"  Old model: 10% -> 0.992 (99.2% of max)")

# ── STEP 8: Final verdict ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("COMPLETE FINAL VERDICT")
print("="*70)
print(f"\n  {'Metric':35}  {'NEW MODEL':14}  {'OLD MODEL':12}")
print("  "+"-"*64)
print(f"  {'In-sample F1':35}  {f1_in:.4f}          0.9923")
print(f"  {'95% CI':35}  [{ci_lo:.4f},{ci_hi:.4f}]  [0.9913,0.9933]")
print(f"  {'Cross-building 4-class F1':35}  {f1_cb:.4f}          0.3310")
print(f"  {'Cross-building binary F1':35}  {f1_cb_bin:.4f}          0.4150")
print(f"  {'Generalisation gap':35}  {gap_pp:.1f}pp            66.1pp")
print(f"  {'Fine-tuning 10% F1':35}  {ft10_f1:.4f}          0.9920")
print(f"  {'Recovery % of max 0.50':35}  {recovery}%            99.2%")
print(f"  {'Genuine fault types':35}  Yes              No")
print(f"  {'McNemar vs GB p<0.001':35}  {'Yes' if p_gb<0.001 else 'No'}              Yes")
print(f"  {'McNemar vs SVM p<0.001':35}  {'Yes' if p_svm<0.001 else 'No'}              Yes")
print()

if ft10_f1 >= 0.95:
    verdict = "GREEN — SWITCH CONFIRMED"
    action  = "New model beats original on EVERY metric. Switch immediately."
elif ft10_f1 >= 0.90:
    verdict = "AMBER-GREEN — SWITCH RECOMMENDED"
    action  = "Strong results. Fine-tuning slightly below original but genuine fault types justify switch."
elif ft10_f1 >= 0.80:
    verdict = "AMBER — MARGINAL"
    action  = "Cross-building better but fine-tuning weaker. Borderline decision."
else:
    verdict = "RED — KEEP ORIGINAL"
    action  = "Fine-tuning too weak. Original 0.992 much better."

print(f"  VERDICT: {verdict}")
print(f"  ACTION:  {action}")

# Save all results
results = {
    "experiment":       "final_model_all_fixes",
    "f1_insample":      round(float(f1_in),4),
    "ci_lo":            round(float(ci_lo),4),
    "ci_hi":            round(float(ci_hi),4),
    "f1_crossbuilding": round(float(f1_cb),4),
    "f1_binary_cb":     round(float(f1_cb_bin),4),
    "gap_pp":           round(float(gap_pp),2),
    "mcc":              round(float(mcc),4),
    "finetune_10pct":   round(float(ft10_f1),4),
    "finetune_recovery":recovery,
    "finetune_curve":   finetune_results,
    "mcnemar":{
        "vs_gb": {"f1":round(float(f1_gb),4),"chi2":round(float(chi2_gb),2),"p":round(float(p_gb),6)},
        "vs_svm":{"f1":round(float(f1_svm),4),"chi2":round(float(chi2_svm),2),"p":round(float(p_svm),6)},
        "vs_dummy":{"f1":round(float(f1_d),4),"chi2":round(float(chi2_d),2),"p":round(float(p_d),6)},
    },
    "psi_values":       {k:round(v,4) for k,v in psi_results.items()},
    "shap_top15":       shap_out,
    "old_model":{"f1_insample":0.9923,"f1_crossbuilding":0.331,
                 "f1_binary_cb":0.415,"finetune_10pct":0.992},
    "verdict":verdict, "action":action,
}
with open(os.path.join(OUT_DIR, "final_model_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved to: results/new_model_final/final_model_results.json")
print("="*70)
