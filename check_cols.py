"""Quick check — print exact column names from both datasets"""
import pandas as pd

ASHRAE_DIR = r'C:\Users\hrslp\Desktop\thesis\data'
SDAHU_DIR  = r'C:\Users\hrslp\Desktop\thesis\data\sdahu'

ashrae = pd.read_csv(f'{ASHRAE_DIR}/MZVAV-1.csv', nrows=3)
sdahu  = pd.read_csv(f'{SDAHU_DIR}/AHU_annual.csv', nrows=3)

print("ASHRAE columns:")
for c in sorted(ashrae.columns):
    print(f"  '{c}'")

print("\nSD-AHU columns (after rename):")
COL_MAP = {
    'SF_CS':'AHU: Supply Air Fan Speed Control Signal',
    'SF_SPD':'AHU: Supply Air Fan Speed Position',
    'RF_CS':'AHU: Return Air Fan Speed Control Signal',
    'RF_SPD':'AHU: Return Air Fan Speed Position',
    'SA_TEMP':'AHU: Supply Air Temperature',
    'SA_SP':'AHU: Supply Air Duct Static Pressure',
    'RA_TEMP':'AHU: Return Air Temperature',
    'OA_TEMP':'AHU: Outdoor Air Temperature',
    'MA_TEMP':'AHU: Mixed Air Temperature',
    'CHWC_VLV':'AHU: Cooling Coil Valve Position',
}
sdahu = sdahu.rename(columns=COL_MAP)
for c in sorted(sdahu.columns):
    print(f"  '{c}'")

print("\nShared:")
shared = sorted(set(ashrae.columns) & set(sdahu.columns)
                - {'Datetime','datetime','Fault Detection Ground Truth'})
for c in shared:
    print(f"  '{c}'")
