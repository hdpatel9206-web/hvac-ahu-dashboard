# MASTER RUN SCRIPT - Run all Chapter 4 fixes in order
# Run in PowerShell: python run_all_fixes.py

import subprocess
import sys
import os

SCRIPTS = [
    ("fix1_row_counts.py",        "FIX 1 - Actual row counts for Table 4.1"),
    ("fix2_baselines.py",         "FIX 2 - Baseline model comparison for Table 4.2"),
    ("fix3_shap.py",              "FIX 3 - Real SHAP values for Table 4.4"),
    ("fix4_crossbuilding_psi.py", "FIX 4+5 - Cross-building per-class + full PSI table"),
    ("fix6_sdahu_physics_psi.py", "FIX 6+7 - SD-AHU PSI and physics normalisation PSI"),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 65)
print("CHAPTER 4 VERIFICATION - Running all fixes")
print("=" * 65)

for script, description in SCRIPTS:
    path = os.path.join(BASE_DIR, script)
    if not os.path.exists(path):
        print(f"\n[SKIP] {script} not found")
        print(f"  Expected at: {path}")
        continue

    print(f"\n{'=' * 65}")
    print(f"RUNNING: {description}")
    print(f"{'=' * 65}")

    result = subprocess.run([sys.executable, path])

    if result.returncode != 0:
        print(f"[ERROR] {script} failed")
    else:
        print(f"[DONE] {script} completed")

print("\n" + "=" * 65)
print("ALL FIXES COMPLETE")
print("Results saved to: results folder in thesis directory")
print("Update Chapter 4 tables with the printed values above.")
print("=" * 65)
