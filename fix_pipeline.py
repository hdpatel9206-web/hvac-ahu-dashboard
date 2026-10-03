f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'r', encoding='utf-8')
src = f.read()
f.close()

# Fix 1: add BOILER_STRUCTURAL_NAN_COLS
if 'BOILER_STRUCTURAL_NAN_COLS' not in src:
    src = src.replace('BOILER_COL_MAP = {', 'BOILER_STRUCTURAL_NAN_COLS = ["PM_POW_1", "PM_POW_2"]\n\nBOILER_COL_MAP = {', 1)
    print('Fix1 applied')
else:
    print('Fix1 already present')

# Fix 2: impute missing features before scaling
# Find the X_raw line
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'X_raw' in line and 'df_feat' in line and 'values' in line:
        print(f'Found X_raw at line {i}: {repr(line)}')

f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'w', encoding='utf-8')
f.write(src)
f.close()
print('Done')
