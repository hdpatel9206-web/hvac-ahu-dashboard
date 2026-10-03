"""
config.py
---------
Central configuration for the HVAC Predictive Maintenance Dashboard.
All constants, paths, colour mappings, and UI settings live here so
every other module imports from a single source of truth.
"""

from pathlib import Path

# ── File paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")

DATA_PATH         = DATA_DIR / "latest_predictions.csv"
MODEL_PATH        = MODEL_DIR / "hvac_multifault_rf_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "hvac_multifault_feature_cols.pkl"

REQUIRED_FILES = [DATA_PATH, MODEL_PATH, FEATURE_COLS_PATH]

# ── Fault class definitions ──────────────────────────────────────────────────
LABEL_MAP: dict[int, str] = {
    0: "Healthy",
    1: "Bearing Wear",
    2: "Overheating / Fouling",
    3: "Power Inefficiency",
}

# Columns that hold the per-class probabilities in the CSV
PROB_COL_MAP: dict[int, str] = {
    0: "Prob_Healthy",
    1: "Prob_Bearing Wear",
    2: "Prob_Overheating/Fouling",
    3: "Prob_Power Inefficiency",
}

# ── Fault visual styling ─────────────────────────────────────────────────────
# Maps fault name → (streamlit status function name, hex colour for charts)
FAULT_STYLE: dict[str, tuple[str, str]] = {
    "Healthy":               ("success", "#2ecc71"),
    "Bearing Wear":          ("warning", "#f39c12"),
    "Overheating / Fouling": ("error",   "#e74c3c"),
    "Power Inefficiency":    ("warning", "#e67e22"),
}

FAULT_COLOURS: list[str] = [v[1] for v in FAULT_STYLE.values()]

# ── Maintenance recommendations ──────────────────────────────────────────────
RECOMMENDATIONS: dict[str, str] = {
    "Healthy": (
        "✅ **No action required.** "
        "System is operating within normal parameters. "
        "Continue scheduled routine inspection every 30 days."
    ),
    "Bearing Wear": (
        "⚠️ **Schedule bearing inspection within 7 days.** "
        "Elevated vibration signatures suggest progressive bearing degradation. "
        "Lubricate or replace bearings to prevent compressor failure. "
        "Monitor vibration trend daily until resolved."
    ),
    "Overheating / Fouling": (
        "🚨 **Immediate action required.** "
        "Thermal anomaly detected — inspect heat exchange surfaces for fouling or blockage. "
        "Verify coolant flow rate and condenser performance. "
        "Risk of thermal shutdown if unaddressed within 24 hours."
    ),
    "Power Inefficiency": (
        "⚠️ **Investigate within 3 days.** "
        "Abnormal power draw detected relative to load. "
        "Check compressor efficiency ratio, refrigerant charge, and VFD settings. "
        "Potential 15–25% energy waste if unresolved."
    ),
}

# ── Sensor metadata ──────────────────────────────────────────────────────────
# Maps column name → (display label, unit string)
SENSOR_META: dict[str, tuple[str, str]] = {
    "Vibration_Hz":   ("Vibration",   "Hz"),
    "Temperature_C":  ("Temperature", "°C"),
    "Power_kW":       ("Power Draw",  "kW"),
}

# ── SHAP settings ────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 300   # max rows sent to SHAP (performance cap)
SHAP_TOP_N       = 10    # features shown in summary bar chart

# ── Dashboard metadata (shown in footer / about) ─────────────────────────────
THESIS_META = {
    "title":    "Explainable ML for HVAC Chiller Predictive Maintenance",
    "subtitle": "Multi-Fault Classification with SHAP-Based Decision Support",
    "framework": "Design Science Research (Hevner et al., 2004)",
    "model":    "Random Forest Classifier  ·  scikit-learn",
    "xai":      "SHAP TreeExplainer  ·  Lundberg & Lee (2017)",
}
