# HVAC Chiller Predictive Maintenance Dashboard

**Explainable ML for Multi-Fault Detection in Smart Building HVAC Systems**

> Thesis Artefact — Design Science Research prototype
> Model: Random Forest Classifier | XAI: SHAP TreeExplainer
> Reference framework: Hevner et al. (2004)

---

## Research Context

This dashboard is the prototype artefact for the thesis:

> *"How can an explainable machine learning prototype support
> multi-fault predictive maintenance of HVAC chillers in smart
> building environments?"*

It directly demonstrates all four thesis contributions:

| # | Contribution | Dashboard Page |
|---|---|---|
| 1 | Multi-fault classification framework | Fault Analysis |
| 2 | SHAP explainability integration | Explainability |
| 3 | ML + dashboard decision support | Live Prediction |
| 4 | Practical usability for non-technical users | System Overview |

---

## Project Structure

```
app.py                    ← Entry point; routing + sidebar filters
config.py                 ← Constants, paths, labels, colour maps
data_loader.py            ← Cached I/O, filtering, summary stats
model_utils.py            ← Inference, SHAP, per-class metrics
visualizations.py         ← All chart rendering (Matplotlib/Seaborn/SHAP)
pages/
    overview.py           ← Page 1: System Overview (KPIs, trends)
    fault_analysis.py     ← Page 2: Classification metrics, confusion matrix
    explainability.py     ← Page 3: SHAP summary, heatmap, beeswarm
    live_prediction.py    ← Page 4: Real-time prediction + SHAP waterfall
data/
    latest_predictions.csv
models/
    hvac_multifault_rf_model.pkl
    hvac_multifault_feature_cols.pkl
outputs/                  ← Optional: saved static image exports
```

---

## Required Data Files

| File | Description |
|------|-------------|
| `data/latest_predictions.csv` | Prediction output from the model pipeline |
| `models/hvac_multifault_rf_model.pkl` | Trained Random Forest model |
| `models/hvac_multifault_feature_cols.pkl` | Feature column list (training order) |

### Expected CSV columns

```
Timestamp, Actual_Label, Predicted_Label,
Vibration_Hz, Temperature_C, Power_kW,
[... other feature columns ...],
Prob_Healthy, Prob_Bearing Wear,
Prob_Overheating/Fouling, Prob_Power Inefficiency
```

---

## Installation

```bash
pip install streamlit pandas numpy matplotlib seaborn shap scikit-learn joblib
```

## Run

```bash
streamlit run app.py
```

---

## Fault Classes

| Code | Class | Trigger Sensor |
|------|-------|----------------|
| 0 | Healthy | — |
| 1 | Bearing Wear | Vibration (Hz) |
| 2 | Overheating / Fouling | Temperature (°C) |
| 3 | Power Inefficiency | Power Draw (kW) |

---

## Key References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Carvalho, T. P., et al. (2019). A systematic literature review of ML methods applied to predictive maintenance. *Computers & Industrial Engineering, 137*.
- Hevner, A. R., et al. (2004). Design science in information systems research. *MIS Quarterly, 28(1)*.
- Zhao, Y., Li, T., Zhang, X., & Zhang, C. (2019). AI-based fault detection for building energy systems. *Renewable and Sustainable Energy Reviews, 109*.
