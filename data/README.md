# Data

## Source

The primary dataset is the **LBNL / ASHRAE AHU fault detection benchmark**
(Lawrence Berkeley National Laboratory, *LBNL Fault Detection and Diagnostics Datasets*).

## Raw data is not included

Raw sensor data is not redistributed in this repository. To reproduce the experiments:

1. Download the AHU datasets from the LBNL FDD data portal:
   https://faultdetection.lbl.gov/ (the single-duct AHU set is distributed as
   `LBNL_FDD_Dataset_SDAHU`).
2. Place the CSV files under this folder, e.g.

   ```
   data/
   ├── MZVAV-1.csv, MZVAV-2-1.csv, MZVAV-2-2.csv, SZCAV.csv, SZVAV.csv, RTU.csv
   └── sdahu/            # AHU_annual.csv, coi_bias_*_annual.csv, oa_bias_*_annual.csv, ...
   ```

   File names must match the paths referenced in `configs/` and the experiment scripts.

## What is included

Only the one file the dashboard loads:

| File | Purpose |
|---|---|
| `chiller_predictions.csv` | Per-sample predictions of the canonical run (`results/run_20260409_022716_ASHRAE_LBNL`), used by the dashboard pages. Despite the name, these are AHU predictions. |
