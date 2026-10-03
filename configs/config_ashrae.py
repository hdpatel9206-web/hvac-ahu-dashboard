# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG — ASHRAE LBNL DATASET (Current)                     ║
# ║  Run with:                                                   ║
# ║    python run_experiment.py --config configs/config_ashrae.py║
# ╚══════════════════════════════════════════════════════════════╝

# ── Dataset identity ──────────────────────────────────────────
DATASET_NAME = 'ASHRAE_LBNL'

# ── Where the CSV files are (relative to thesis/ root folder) ─
DATA_DIR = 'data'

# ── Training files and their fault class numbers ──────────────
# Format: 'filename.csv': fault_class_number
# Class 0 is always Healthy (assigned automatically)
TRAIN_FILES = {
    'MZVAV-1.csv':   1,    # Fault class 1 = OA sensor bias
    'MZVAV-2-1.csv': 2,    # Fault class 2 = Heating coil valve leak
    'SZCAV.csv':     3,    # Fault class 3 = Control fault
}

# ── Cross-building test files (never used in training) ────────
CROSS_BUILDING_FILES = {
    'MZVAV-2-2.csv': 2,    # Same fault type as MZVAV-2-1 but different site
    'SZVAV.csv':     3,    # Same fault type as SZCAV but different site
}

# ── Ground truth column name in the CSV ───────────────────────
GROUND_TRUTH_COL = 'Fault Detection Ground Truth'

# ── Value in ground truth column that means FAULT ─────────────
FAULT_VALUE = 1

# ── Human-readable label names ────────────────────────────────
LABEL_MAP = {
    0: 'Healthy',
    1: 'Fault-A (OA Sensor)',
    2: 'Fault-B (Valve Leak)',
    3: 'Fault-C (Control)',
}

# ── Feature columns to use from the CSV ───────────────────────
FEATURE_COLS = [
    'AHU: Supply Air Temperature',
    'AHU: Outdoor Air Temperature',
    'AHU: Mixed Air Temperature',
    'AHU: Return Air Temperature',
    'AHU: Supply Air Fan Status',
    'AHU: Supply Air Fan Speed Control Signal',
    'AHU: Cooling Coil Valve Control Signal',
    'AHU: Heating Coil Valve Control Signal',
    'Occupancy Mode Indicator',
]

# ── Derived features: name -> (column_A, column_B) → A minus B ─
DERIVED_FEATURES = {
    'Temp_supply_return_diff':  ('AHU: Supply Air Temperature', 'AHU: Return Air Temperature'),
    'Temp_outdoor_supply_diff': ('AHU: Supply Air Temperature', 'AHU: Outdoor Air Temperature'),
    'Temp_mixed_return_diff':   ('AHU: Mixed Air Temperature',  'AHU: Return Air Temperature'),
    'Valve_diff':               ('AHU: Cooling Coil Valve Control Signal', 'AHU: Heating Coil Valve Control Signal'),
    'Fan_damper_ratio':         ('AHU: Supply Air Fan Speed Control Signal', 'AHU: Outdoor Air Damper Control Signal'),
}

# ── Columns to exclude from model features ────────────────────
EXCLUDE_COLS = [
    'AHU: Supply Air Temperature Set Point',
    'AHU: Supply Air Temperature Heating Set Point',
    'AHU: Supply Air Temperature Cooling Set Point',
    'AHU: Supply Air Duct Static Pressure Set Point',
    'AHU: Supply Air Duct Static Pressure',
    'AHU: Return Air Fan Status',
    'AHU: Return Air Fan Speed Control Signal',
    'AHU: Exhaust Air Damper Control Signal',
]

# ── Rolling window sizes (in number of observations) ──────────
ROLLING_WINDOWS = [10, 30, 60]

# ── Subsample per class to prevent one site dominating ────────
# Set to None to use all data
SUBSAMPLE_PER_CLASS = 30000

# ── Train/test split ratio ────────────────────────────────────
TEST_SIZE = 0.20

# ── Bootstrap iterations for confidence intervals ─────────────
N_BOOTSTRAP = 1000

# ── Include Gradient Boosting as extra baseline? ──────────────
INCLUDE_GRADIENT_BOOSTING = True

# ── Random seed ───────────────────────────────────────────────
RANDOM_STATE = 42
