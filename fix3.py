f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'r', encoding='utf-8')
src = f.read()
f.close()

src = src.replace(
    'BOILER_COL_MAP = {',
    'BOILER_STRUCTURAL_NAN_COLS = ["PM_POW_1", "PM_POW_2"]\n\nBOILER_COL_MAP = {',
    1
)
print('Fix applied')

f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'w', encoding='utf-8')
f.write(src)
f.close()
print('Done')
