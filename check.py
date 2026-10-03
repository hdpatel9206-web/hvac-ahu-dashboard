f = open(r'C:\Users\hrslp\Desktop\thesis\predict_pipeline.py', 'r', encoding='utf-8')
lines = f.readlines()
f.close()
for i, line in enumerate(lines):
    if 'BOILER_STRUCTURAL_NAN_COLS' in line:
        print(f'Line {i+1}: {line.rstrip()}')
