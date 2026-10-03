import pandas as pd
ASHRAE_DIR = r'C:\Users\hrslp\Desktop\thesis\data'
df = pd.read_csv(f'{ASHRAE_DIR}/SZCAV.csv', nrows=3)
df.columns = df.columns.str.strip()
print("SZCAV exact columns:")
for c in sorted(df.columns):
    print(f"  repr: {repr(c)}")
