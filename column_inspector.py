"""
Step 1 — Column Inspector
Universal HVAC FDD Model Registry | M.Sc. Thesis, bbw Hochschule Berlin
Supervisors: Prof. Farshi Hossein (1st), Prof. Dr. Juan Ocampo (2nd)

Usage:
  python column_inspector.py                        # inspects all known data folders
  python column_inspector.py path/to/file.csv       # inspects a single file
  python column_inspector.py path/to/folder/        # inspects all CSVs in a folder
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE = r"C:\Users\hrslp\Desktop\thesis\data"

KNOWN_FOLDERS = {
    "AHU (ASHRAE)": BASE,
    "SD-AHU":       os.path.join(BASE, "sdahu"),
    "FCU":          os.path.join(BASE, "fcu"),
    "Boiler":       os.path.join(BASE, "boiler"),
}

EXCLUDE_COLS = {
    "datetime", "timestamp", "date", "time",
    "fault detection ground truth", "label",
}

EXCEL_ERRORS = {"#VALUE!", "#REF!", "#DIV/0!", "#N/A"}

EVAL_CAP     = 15_000   # rows loaded for inspection (matches evaluation cap)

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_load(path: str) -> pd.DataFrame:
    """Load up to EVAL_CAP rows, strip column names, replace Excel errors."""
    df = pd.read_csv(path, nrows=EVAL_CAP, low_memory=False)
    df.columns = df.columns.str.strip()
    # Replace all known Excel error strings with NaN
    df.replace(list(EXCEL_ERRORS), pd.NA, inplace=True)
    return df


def feature_cols(df: pd.DataFrame) -> list:
    """Return non-excluded columns (case-insensitive match on exclusion list)."""
    return [c for c in df.columns if c.lower() not in EXCLUDE_COLS]


def inspect_file(path: str) -> dict:
    """Return a dict with column info for one CSV file."""
    try:
        df = safe_load(path)
        fcols = feature_cols(df)
        dtypes = {c: str(df[c].dtype) for c in fcols}
        nan_pct = {c: round(df[c].isna().mean() * 100, 1) for c in fcols}
        return {
            "status":    "ok",
            "rows":      len(df),
            "all_cols":  list(df.columns),
            "feat_cols": fcols,
            "dtypes":    dtypes,
            "nan_pct":   nan_pct,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def intersection(col_lists: list[list]) -> list:
    """Return sorted intersection of multiple column lists."""
    if not col_lists:
        return []
    result = set(col_lists[0])
    for lst in col_lists[1:]:
        result &= set(lst)
    return sorted(result)


def union(col_lists: list[list]) -> list:
    result = set()
    for lst in col_lists:
        result |= set(lst)
    return sorted(result)

# ── Reporters ─────────────────────────────────────────────────────────────────

def report_file(path: str):
    print(f"\n{'='*70}")
    print(f"FILE: {path}")
    info = inspect_file(path)
    if info["status"] == "error":
        print(f"  ERROR: {info['message']}")
        return info
    print(f"  Rows loaded (capped at {EVAL_CAP:,}): {info['rows']:,}")
    print(f"  Total columns      : {len(info['all_cols'])}")
    print(f"  Feature columns    : {len(info['feat_cols'])}")
    print()
    print("  FEATURE COLUMNS (name | dtype | %NaN):")
    for c in info["feat_cols"]:
        print(f"    {c:<50}  {info['dtypes'][c]:<10}  {info['nan_pct'][c]:>5}%")
    excluded = [c for c in info["all_cols"] if c not in info["feat_cols"]]
    if excluded:
        print(f"\n  EXCLUDED (id/label cols): {excluded}")
    return info


def report_folder(label: str, folder: str):
    print(f"\n{'#'*70}")
    print(f"FOLDER: {label}  →  {folder}")
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csvs:
        print("  No CSV files found.")
        return

    print(f"  {len(csvs)} CSV file(s) found.\n")
    all_feat_cols = []
    file_results = {}

    for path in csvs:
        fname = os.path.basename(path)
        info = inspect_file(path)
        file_results[fname] = info
        if info["status"] == "ok":
            all_feat_cols.append(info["feat_cols"])
            rows_str = f"{info['rows']:>7,} rows"
            ncols    = len(info["feat_cols"])
            print(f"  {fname:<50}  {rows_str}  {ncols} feature cols")
        else:
            print(f"  {fname:<50}  ERROR: {info['message']}")

    if len(all_feat_cols) < 2:
        print("\n  (Only one file — skipping intersection analysis.)")
        return

    shared  = intersection(all_feat_cols)
    all_u   = union(all_feat_cols)
    missing_map = {}   # col → list of files that lack it
    for col in all_u:
        absent = [os.path.basename(csvs[i]) for i, lst in enumerate(all_feat_cols)
                  if col not in lst]
        if absent:
            missing_map[col] = absent

    print(f"\n  ── Intersection across all {len(csvs)} files ──")
    print(f"  Shared feature columns  : {len(shared)}")
    print(f"  Total union of columns  : {len(all_u)}")
    print(f"  Columns missing in ≥1 file: {len(missing_map)}")

    print(f"\n  SHARED COLUMNS ({len(shared)}):")
    for c in shared:
        print(f"    {c}")

    if missing_map:
        print(f"\n  COLUMNS MISSING IN SOME FILES:")
        for col, absent_files in sorted(missing_map.items()):
            print(f"    {col}")
            for f in absent_files:
                print(f"      ↳ missing in: {f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        # Default: inspect all known folders
        print("Universal HVAC FDD — Column Inspector")
        print("Running on all configured data folders...\n")
        for label, folder in KNOWN_FOLDERS.items():
            if os.path.isdir(folder):
                report_folder(label, folder)
            else:
                print(f"\n[SKIP] {label}: folder not found → {folder}")

    elif len(args) == 1:
        target = args[0]
        if os.path.isfile(target) and target.endswith(".csv"):
            report_file(target)
        elif os.path.isdir(target):
            report_folder(os.path.basename(target), target)
        else:
            print(f"ERROR: '{target}' is not a CSV file or directory.")
            sys.exit(1)

    else:
        # Multiple files passed — show individual reports + combined intersection
        print(f"Inspecting {len(args)} files + combined intersection...\n")
        all_feat_cols = []
        for path in args:
            info = report_file(path)
            if info.get("status") == "ok":
                all_feat_cols.append(info["feat_cols"])

        if len(all_feat_cols) >= 2:
            shared = intersection(all_feat_cols)
            print(f"\n{'='*70}")
            print(f"COMBINED INTERSECTION across {len(args)} files: {len(shared)} columns")
            for c in shared:
                print(f"  {c}")

    print(f"\n{'='*70}")
    print("Column inspection complete.")


if __name__ == "__main__":
    main()
