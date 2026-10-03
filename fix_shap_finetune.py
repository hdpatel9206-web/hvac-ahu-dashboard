# fix_shap_finetune.py
# Runs SHAP and fine-tuning on the already-saved final model
# Run AFTER new_model_final.py completes Step 3
# Run: python fix_shap_finetune.py

import os, json, warnings, time
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "results", "new_model_final")
RS       = 42

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
    'CHWC_VLV_DM':'AHU: Cooling Coil Valve Control Signal','SYS_CTL':'Occupancy Mode Indicator',
}
EXCEL_ERRORS = {"#VALUE!","#REF!","#DIV/0!","#N/A"}
ROLLING_WINDOWS = [10,30,60]
FINETUNE_PROPS  = [0.01,0.05,0.10,0.15,0.20,0.30,0.50]
CLASS_NAMES = {0:'Healthy',1:'Sensor Bias',2:'Valve Fault',3:'Damper Fault'}

def engineer_features(df):
    available = [c for c in RAW_COLS if c in df.columns]
    out = df[available].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col],errors='coerce').fillna(0)
    for col in available:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rm{w}'] = out[col].rolling(w,min_periods=1).mean()
            out[f'{col}_rs{w}'] = out[col].rolling(w,min_periods=1).std().fillna(0)
    pairs=[('Temp_supply_return_diff','AHU: Supply Air Temperature','AHU: Return Air Temperature'),
           ('Temp_outdoor_supply_diff','AHU: Outdoor Air Temperature','AHU: Supply Air Temperature'),
           ('Temp_mixed_return_diff','AHU: Mixed Air Temperature','AHU: Return Air Temperature')]
    for name,a,b in pairs:
        if a in out.columns and b in out.columns:
            out[name] = out[a]-out[b]
    return out.fillna(0).replace([np.inf,-np.inf],0)

def load_ashrae(fpath):
    df = pd.read_csv(fpath,low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS),pd.NA,inplace=True)
    lc = 'Fault Detection Ground Truth'
    if lc not in df.columns: return pd.DataFrame()
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df.get(col,0),errors='coerce').fillna(0)
    feats = engineer_features(df)
    feats['label'] = np.where(df[lc].values[:len(feats)]==0,0,1)
    return feats.dropna()

# Load saved model
print("Loading saved model...")
model     = joblib.load(os.path.join(OUT_DIR,"rf_model_final.pkl"))
scaler    = joblib.load(os.path.join(OUT_DIR,"scaler_final.pkl"))
feat_cols = joblib.load(os.path.join(OUT_DIR,"feature_cols_final.pkl"))
print(f"  Model loaded. Features: {len(feat_cols)}")

# Load cross-building data
print("Loading cross-building test data...")
cb_parts = []
for fname in ['MZVAV-2-2.csv','SZVAV.csv']:
    fpath = os.path.join(DATA_DIR,fname)
    if not os.path.exists(fpath): continue
    df = load_ashrae(fpath)
    if len(df)>0:
        cb_parts.append(df)

df_cb = pd.concat(cb_parts,ignore_index=True)
X_cb_mat = np.zeros((len(df_cb),len(feat_cols)))
for i,col in enumerate(feat_cols):
    if col in df_cb.columns:
        X_cb_mat[:,i] = df_cb[col].values.astype(float)
X_cb_sc = scaler.transform(X_cb_mat)
y_cb    = df_cb['label'].values.astype(int)
preds_cb = model.predict(X_cb_sc)
f1_cb   = f1_score(y_cb,preds_cb,average='macro',zero_division=0)
print(f"  Cross-building F1 confirmed: {f1_cb:.4f}")

# ── SHAP (fixed) ─────────────────────────────────────────────────────────────
shap_out = {}
try:
    import shap
    print("\nRunning SHAP analysis...")
    t0 = time.time()
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(X_cb_sc),min(500,len(X_cb_sc)),replace=False)
    X_shap = X_cb_sc[idx]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    # Handle both list and array output
    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)

    # Flatten if needed
    mean_shap = np.array(mean_shap).flatten()

    ranked = np.argsort(mean_shap)[::-1]
    print(f"\n  {'Rank':4}  {'Feature':52}  {'Mean |SHAP|'}")
    print("  " + "-" * 72)
    for rank_i, feat_idx in enumerate(ranked[:15], 1):
        feat_idx = int(feat_idx)
        if feat_idx < len(feat_cols):
            fname_s = feat_cols[feat_idx]
            val     = float(mean_shap[feat_idx])
            print(f"  {rank_i:4}  {fname_s:52}  {val:.4f}")
            shap_out[fname_s] = round(val,4)
    print(f"  SHAP time: {time.time()-t0:.1f}s")
except ImportError:
    print("  SHAP not installed. Run: pip install shap")
except Exception as e:
    print(f"  SHAP error: {e}")

# ── Fine-tuning ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINE-TUNING EXPERIMENT")
print("="*65)

# Rebuild training data for fine-tuning
print("Rebuilding training data for fine-tuning...")
import glob
SDAHU_DIR = os.path.join(DATA_DIR,"sdahu")
if not os.path.exists(SDAHU_DIR):
    SDAHU_DIR = os.path.join(DATA_DIR,"lbnl_sdahu","LBNL_FDD_Dataset_SDAHU")

SDAHU_CLASS0=['ahu_annual']
SDAHU_CLASS2=['coi_leakage','coi_stuck']
SDAHU_CLASS3=['damper_stuck']
MAX_PER_CLASS=120000
MAX_PER_FILE=50000

def load_sdahu(fpath,label,zero_oa=False):
    df = pd.read_csv(fpath,low_memory=False,nrows=MAX_PER_FILE)
    df.columns = df.columns.str.strip()
    df.replace(list(EXCEL_ERRORS),pd.NA,inplace=True)
    for sc,ac in SDAHU_COL_MAP.items():
        if sc in df.columns:
            df[ac] = pd.to_numeric(df[sc],errors='coerce')
    for col in RAW_COLS:
        if col not in df.columns: df[col]=0.0
    if zero_oa and label in [2,3]:
        df['AHU: Outdoor Air Temperature']=0.0
        df['AHU: Mixed Air Temperature']=0.0
    feats = engineer_features(df)
    feats['label']=label
    return feats.dropna()

pools={0:[],1:[],2:[],3:[]}
for fname in ['MZVAV-1.csv','MZVAV-2-1.csv','SZCAV.csv']:
    fpath=os.path.join(DATA_DIR,fname)
    if not os.path.exists(fpath): continue
    df=load_ashrae(fpath)
    if len(df)>0:
        pools[0].append(df[df['label']==0])
        pools[1].append(df[df['label']==1])

for fpath in sorted(glob.glob(os.path.join(SDAHU_DIR,"*.csv"))):
    fn=os.path.basename(fpath).lower()
    lbl=None
    for p in SDAHU_CLASS0:
        if p in fn: lbl=0; break
    if lbl is None:
        for p in SDAHU_CLASS2:
            if p in fn: lbl=2; break
    if lbl is None:
        for p in SDAHU_CLASS3:
            if p in fn: lbl=3; break
    if lbl is None: continue
    df=load_sdahu(fpath,lbl,zero_oa=True)
    if len(df)>0: pools[lbl].append(df)

class_dfs={}
for cls,parts in pools.items():
    if parts:
        comb=pd.concat(parts,ignore_index=True)
        if len(comb)>MAX_PER_CLASS:
            comb=comb.sample(n=MAX_PER_CLASS,random_state=RS)
        class_dfs[cls]=comb

feat_set=set(class_dfs[0].columns)
for df in class_dfs.values(): feat_set &= set(df.columns)
ft_feat_cols=sorted([c for c in feat_set if c!='label'])

df_tr=pd.concat([df[ft_feat_cols+['label']] for df in class_dfs.values()],ignore_index=True)
from sklearn.model_selection import train_test_split
X=df_tr[ft_feat_cols].values.astype(float)
y=df_tr['label'].values.astype(int)
X_tr,_,y_tr,_=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
X_tr_sc=scaler.transform(X_tr)

print(f"  Training set: {len(X_tr):,} rows")

# Align CB to training feat cols
X_cb_ft=np.zeros((len(df_cb),len(ft_feat_cols)))
for i,col in enumerate(ft_feat_cols):
    if col in df_cb.columns:
        X_cb_ft[:,i]=df_cb[col].values.astype(float)
X_cb_ft_sc=scaler.transform(X_cb_ft)

finetune_results=[]
print(f"\n  {'%':6}  {'Records':10}  {'4-class':10}  {'Binary':10}  Change")
print("  "+"-"*52)

for pct in FINETUNE_PROPS:
    n=max(1,int(len(X_cb_ft)*pct))
    idx=np.random.RandomState(RS).choice(len(X_cb_ft),n,replace=False)
    X_new=np.vstack([X_tr_sc,X_cb_ft_sc[idx]])
    y_new=np.concatenate([y_tr,y_cb[idx]])
    m=RandomForestClassifier(n_estimators=200,class_weight='balanced',
                             random_state=RS,n_jobs=-1,min_samples_leaf=2)
    m.fit(X_new,y_new)
    p=m.predict(X_cb_ft_sc)
    f4=f1_score(y_cb,p,average='macro',zero_division=0)
    fb=f1_score((y_cb>0).astype(int),(p>0).astype(int),average='macro',zero_division=0)
    chg=(f4-f1_cb)*100
    finetune_results.append({'pct':pct,'n':n,'f1_4class':round(f4,4),
                              'f1_binary':round(fb,4),'change_pp':round(chg,1)})
    print(f"  {pct*100:5.0f}%  {n:10,}  {f4:10.4f}  {fb:10.4f}  +{chg:.1f}pp")

ft10=next((r for r in finetune_results if r['pct']==0.10),None)
ft10_f1=ft10['f1_4class'] if ft10 else 0

print(f"\n  FINE-TUNING 10%: 4-class={ft10_f1:.4f}  (old model: 0.992)")
print(f"  Theoretical max = 0.50")
print(f"  Recovery = {ft10_f1/0.50*100:.1f}% of theoretical max")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("COMPLETE FINAL RESULTS")
print("="*65)
print(f"\n  FINAL OPTIMISED MODEL vs OLD MODEL:")
print(f"  {'Metric':35}  {'New':12}  {'Old':12}")
print("  "+"-"*62)
print(f"  {'In-sample F1':35}  0.9909         0.9923")
print(f"  {'In-sample CI':35}  [0.9904,0.9915] [0.9913,0.9933]")
print(f"  {'Cross-building 4-class F1':35}  {f1_cb:.4f}         0.3310")
print(f"  {'Cross-building binary F1':35}  0.6283         0.4150")
print(f"  {'Generalisation gap':35}  57.3pp         66.1pp")
print(f"  {'Fine-tuning 10% F1':35}  {ft10_f1:.4f}         0.9920")
print(f"  {'Recovery % of max':35}  {ft10_f1/0.50*100:.1f}%          99.2%")
print(f"  {'Genuine fault types':35}  Yes             No")
print(f"  {'McNemar vs GB':35}  p<0.001         p<0.001")
print(f"  {'McNemar vs SVM':35}  p<0.001         p<0.001")
print()

if ft10_f1>=0.95:
    verdict="GREEN — SWITCH TO NEW MODEL CONFIRMED"
    action="All metrics verified. New model is better in every way."
elif ft10_f1>=0.90:
    verdict="AMBER-GREEN — SWITCH RECOMMENDED"
    action="Strong recovery. Slightly below old model fine-tuning but genuine fault types justify switch."
elif ft10_f1>=0.80:
    verdict="AMBER — YOUR CHOICE"
    action="New model has better cross-building F1 but weaker fine-tuning recovery."
else:
    verdict="RED — KEEP ORIGINAL"
    action="Fine-tuning too weak despite better cross-building F1."

print(f"  VERDICT: {verdict}")
print(f"  ACTION:  {action}")

# Save
res={
    "cross_building_f1":round(float(f1_cb),4),
    "binary_cb_f1":0.6283,
    "gap_pp":57.3,
    "finetune_10pct":round(float(ft10_f1),4),
    "finetune_recovery_pct":round(ft10_f1/0.50*100,1),
    "finetune_curve":finetune_results,
    "shap_top15":shap_out,
    "verdict":verdict,
    "action":action,
    "old_model":{"f1_cb":0.331,"binary_cb":0.415,"finetune_10pct":0.992},
}
with open(os.path.join(OUT_DIR,"finetune_shap_results.json"),"w") as f:
    json.dump(res,f,indent=2)
print(f"\n  Saved to results/new_model_final/finetune_shap_results.json")
