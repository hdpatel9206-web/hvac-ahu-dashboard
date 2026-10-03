# ============================================================
#  CHILLER EXPERIMENT CONFIG — ASHRAE RP-1043
#  Dataset: Figshare Chiller Operation Data Set
#  Path: data/chiller/
# ============================================================

import os
from pathlib import Path

DATASET_NAME   = "CHILLER_RP1043"
BASE_DIR       = Path(r"C:\Users\hrslp\Desktop\thesis")
DATA_DIR       = BASE_DIR / "data" / "chiller"

# ── Raw sensor feature columns (from RP-1043, 65 variables) ──
RAW_FEATURE_COLS = [
    'TWE_set','TEI','TWEI','TEO','TWEO','TCI','TWCI','TCO','TWCO',
    'TSI','TSO','TBI','TBO',
    'Cond Tons','Cooling Tons','Shared Cond Tons','Cond Energy Balance',
    'Evap Tons','Shared Evap Tons','Building Tons','Evap Energy Balance',
    'kW','COP','kW/Ton','FWC','FWE','TEA','TCA',
    'TRE','PRE','TRC','PRC','TRC_sub','T_suc','Tsh_suc',
    'TR_dis','Tsh_dis','P_lift','Amps','RLA%',
    'Heat Balance (kW)','Heat Balance%','Tolerance%',
    'TO_sump','TO_feed','PO_feed','PO_net',
    'TWCD','TWED','VSS','VSL','VH','VM','VC','VE','VW',
    'TWI','TWO','THI','THO','FWW','FWH','FWB'
]

# ── Rolling windows ──
ROLLING_WINDOWS = [10, 30, 60]

# ── Derived features (temperature and pressure differentials) ──
DERIVED_FEATURES = {
    'Temp_evap_diff':      ('TEO', 'TEI'),
    'Temp_cond_diff':      ('TCO', 'TCI'),
    'Temp_superheat_diff': ('Tsh_dis', 'Tsh_suc'),
    'Pressure_lift':       ('PRC', 'PRE'),
    'COP_kW_ratio':        ('COP', 'kW'),
}

# ── Fault folders and their class labels ──
# Class 0 = Healthy (from Benchmark Tests folder)
# Classes 1-9 = fault types
FAULT_MAP = {
    "Benchmark Tests":              0,   # Healthy
    "Condenser fouling":            1,
    "Defective Pilot Valve":        2,
    "Excess oil":                   3,
    "Refrigerant leak":             4,
    "Refrigerant overcharge":       5,
    "Reduced condenser water flow": 6,
    "Reduced evaporator water flow":7,
    "Non-condensables in refrigerant": 8,
}

# For thesis — use 4 main classes to match AHU experiment
# Healthy + 3 most common chiller faults
FAULT_MAP_SIMPLIFIED = {
    "Benchmark Tests":   0,  # Healthy
    "Condenser fouling": 1,  # Most common — matches thesis narrative
    "Refrigerant leak":  2,  # Critical safety fault
    "Excess oil":        3,  # Common operational fault
}

LABEL_MAP = {
    0: "Healthy",
    1: "Condenser Fouling",
    2: "Refrigerant Leak",
    3: "Excess Oil",
}

# ── Training parameters ──
SUBSAMPLE_PER_CLASS = 5000   # RP-1043 is smaller than ASHRAE AHU
TEST_SIZE           = 0.2
RANDOM_STATE        = 42
N_BOOTSTRAP         = 1000

# ── Time column ──
TIME_COL = 'Time'   # First column — seconds elapsed

# ── File extension ──
FILE_EXT = '.xls'
