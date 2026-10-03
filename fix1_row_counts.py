"""
FIX 1 — Actual Row Counts for All AHU Files
Run in PowerShell: python fix1_row_counts.py
"""

import pandas as pd
import os

BASE = r"C:\Users\hrslp\Desktop\thesis\data"

FILES = {
    "MZVAV-1.csv":   "Training",
    "MZVAV-2-1.csv": "Training",
    "SZCAV.csv":     "Training",
    "MZVAV-2-2.csv": "Test (withheld)",
    "SZVAV.csv":     "Test (withheld)",
}

print("=" * 65)
print("AHU FILE ROW COUNTS — for Table 4.1 in Chapter 4")
print("=" * 65)

total_train = 0
total_test  = 0

for fname, role in FILES.items():
    path = os.path.join(BASE, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        rows = len(df)
        cols = len([c for c in df.columns if c not in
                    ["Datetime", "Fault Detection Ground Truth"]])
        print(f"  {fname:20s}  {rows:8,} rows  {cols:3} feature cols  [{role}]")
        if role == "Training":
            total_train += rows
        else:
            total_test += rows
    else:
        print(f"  {fname:20s}  FILE NOT FOUND")

print("-" * 65)
print(f"  Total training rows : {total_train:,}")
print(f"  Total test rows     : {total_test:,}")
print(f"  Grand total         : {total_train + total_test:,}")
print("=" * 65)
print("\nCopy these numbers into Table 4.1 in Chapter 4.")
