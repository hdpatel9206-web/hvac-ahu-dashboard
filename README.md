# HVAC AHU Fault Detection — Explainable ML with Cross-Building Validation

**MSc Thesis · Smart Building Technologies · bbw Hochschule Berlin (2026)**
*Multi-Fault Detection and Diagnosis for Smart Building AHUs: An Explainable Machine Learning Approach with Cross-Building Validation*

Most fault-detection models for building HVAC are evaluated on the same building they were trained on. This project measures what happens when they are moved to a different building, explains why they fail, and shows how little target-building data is needed to recover.

[**▶ Live dashboard**](LINK_TO_STREAMLIT_APP) · [Thesis abstract](LINK_TO_ABSTRACT_PDF)

---

## Key results

| Metric | Value |
|---|---|
| In-sample macro-F1 (Random Forest, 4 classes) | **0.9923** (95% CI 0.9913–0.9933) · MCC 0.9827 |
| Cross-building macro-F1 (no adaptation) | **0.331** |
| Generalization gap | **66.13 percentage points** |
| Binary F1, 10% target-building adaptation | **0.415 → 0.992** |
| Macro-F1, 10% target-building adaptation | 0.312 → 0.744 |
| Leave-One-Building-Out mean macro-F1 | 0.430 ± 0.123 |

**What drives the gap.** Population Stability Index (PSI) analysis on raw sensor channels shows the shift is concentrated in a few signals. Fan speed has PSI 2.63 and reflects commissioning differences. Outdoor-air temperature has PSI 2.45 and reflects both climate and control logic. Several other channels also fall in the red zone.

**It is not a data-volume problem.** Across 11 training proportions, cross-building F1 stays within 0.498–0.559 (slope ≈ −0.010). More data from the source building does not fix a distribution shift.

**Real buildings are harsher than benchmarks.** The maximum PSI observed on real BMS data was 50.66, against 2.94 on the benchmark.

---

## What's in this repository

- **Fault classifier.** Random Forest on 70 engineered features (11 raw, 54 rolling-window, 5 derived), detecting healthy operation, OA sensor bias, valve leakage and control faults.
- **Explainability** — SHAP global feature ranking from the canonical run (top feature: outdoor-air temperature, mean |SHAP| 0.084).
- **Drift gate.** PSI computed on raw, pre-scaling values against a training reference, with GREEN / AMBER / RED deployment status.
- **Cross-building evaluation.** LOBO cross-validation, few-shot adaptation curves (5–50% target data), and equipment-category transfer (AHU, FCU, RTU, boiler).
- **External validation.** Five BMS datasets from South Korea and Ireland, plus RTU validation (USA).
- **Interactive Streamlit dashboard** for exploring predictions, SHAP explanations and drift status.

## Quick start

```bash
git clone https://github.com/hdpatel9206-web/hvac-ahu-dashboard.git
cd hvac-ahu-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Thesis experiments ran on Python 3.12; dashboard tested on Python 3.14. Results are reproducible with `random_state=42`; canonical run: `run_20260409_022716_ASHRAE_LBNL`.

## Data

The primary dataset is the **LBNL / ASHRAE AHU fault detection benchmark**. Raw data is not redistributed here. Download it from the original source and place it in `data/` (see `data/README.md`).

## Limitations

- In-sample scores do not indicate deployment performance. Check the cross-building numbers first.
- AHU→RTU and AHU→boiler transfer did not recover with adaptation (macro-F1 0.24 and 0.16). Those equipment types needed specialist models under the tested conditions.
- PSI deployment gate: GREEN < 0.20 (deploy), AMBER 0.20–0.50 (collect adaptation data), RED > 0.50 (adapt before deployment). Stricter operational thresholds (e.g. GREEN < 0.10) may suit live monitoring but were not evaluated in this study.
- Live single-sample prediction is not included in this release; the canonical 4-class model (119 MB) exceeds GitHub's file limit.

## Citation

```
Patel, H. (2026). Multi-Fault Detection and Diagnosis for Smart Building AHUs:
An Explainable Machine Learning Approach with Cross-Building Validation.
MSc Thesis, bbw Hochschule Berlin.
```

Supervisors: Prof. Farshi Hossein, Prof. Dr. Juan Ocampo

## Contact

**Harshil Patel** · [LinkedIn](LINK_TO_LINKEDIN) · Berlin, Germany
