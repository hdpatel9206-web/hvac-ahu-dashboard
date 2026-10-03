f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'r', encoding='utf-8')
src = f.read()
f.close()

old = '    X_raw    = df_feat[available].values'
new = '''    if missing:
        for col in missing:
            if col in reg["ref_df"].columns:
                fill_val = pd.to_numeric(reg["ref_df"][col], errors="coerce").mean()
            else:
                fill_val = 0.0
            df_feat = df_feat.copy()
            df_feat[col] = fill_val
    df_feat = df_feat[feat_cols]
    X_raw = df_feat.values.astype(float)'''

if old in src:
    src = src.replace(old, new, 1)
    print('Fix2 applied')
else:
    print('ERROR: target not found')

f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'w', encoding='utf-8')
f.write(src)
f.close()
print('Done')
