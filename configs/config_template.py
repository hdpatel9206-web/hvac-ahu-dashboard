# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG TEMPLATE — FOR ANY NEW DATASET                      ║
# ║                                                              ║
# ║  HOW TO USE THIS FILE:                                       ║
# ║  1. Copy this file and rename it, e.g. config_dataset2.py   ║
# ║  2. Put your new CSV files in data/your_dataset_folder/      ║
# ║  3. Fill in every section below                              ║
# ║  4. Run:                                                     ║
# ║       python run_experiment.py --config configs/config_new.py║
# ╚══════════════════════════════════════════════════════════════╝

# ── STEP 1: Give your dataset a name ─────────────────────────
# This name appears in all figures and the results folder name
DATASET_NAME = 'MY_NEW_DATASET'   # <-- change this

# ── STEP 2: Where are your CSV files? ────────────────────────
# Path relative to the thesis/ root folder
# Example: if files are in thesis/data/new_dataset/ → use 'data/new_dataset'
# Example: if files are directly in thesis/data/    → use 'data'
DATA_DIR = 'data/new_dataset'   # <-- change this

# ── STEP 3: Training files ────────────────────────────────────
# List every CSV file used for training
# The number is the fault CLASS assigned to that file's faults
# Class 0 = Healthy is ALWAYS assigned automatically to non-fault rows
# You can have as many or as few files as you want
# Example with 2 training files:
TRAIN_FILES = {
    'site_A_training.csv': 1,   # fault class 1 for this file
    'site_B_training.csv': 2,   # fault class 2 for this file
    # add more files here if needed
}

# ── STEP 4: Cross-building test files ────────────────────────
# Files NEVER used in training — used only to test generalisation
# Leave empty dict {} if you only have one site and no cross-building test
CROSS_BUILDING_FILES = {
    'site_C_test.csv': 1,   # same fault class as site A but different building
    # add more files here if needed
}

# ── STEP 5: What is the ground truth column called? ───────────
# Open your CSV in Excel and find the column that says 0=normal, 1=fault
# Copy the exact column name here
GROUND_TRUTH_COL = 'Fault_Label'   # <-- change to your column name

# ── STEP 6: What value means FAULT in that column? ────────────
# Usually 1 means fault. Change if your dataset uses different values.
FAULT_VALUE = 1

# ── STEP 7: Human-readable names for each class ───────────────
# Match these to the fault class numbers you assigned above
LABEL_MAP = {
    0: 'Healthy',
    1: 'Fault Type 1',   # <-- describe what fault this is
    2: 'Fault Type 2',   # <-- describe what fault this is
    # add more if you have more fault classes
}

# ── STEP 8: Which columns are your sensor features? ───────────
# List the column names from your CSV that contain sensor readings
# DO NOT include the timestamp column or ground truth column here
FEATURE_COLS = [
    'Temperature_Supply',   # <-- replace with your actual column names
    'Temperature_Return',
    'Fan_Speed',
    'Power_kW',
    'Valve_Position',
    # add more sensor columns here
]

# ── STEP 9: Derived features (optional) ──────────────────────
# These are new features calculated by subtracting one column from another
# Format: 'new_feature_name': ('column_A', 'column_B')  → result = A - B
# Leave empty {} if you don't want any derived features
DERIVED_FEATURES = {
    'Temp_differential': ('Temperature_Supply', 'Temperature_Return'),
    # add more derived features here if needed
}

# ── STEP 10: Columns to exclude ───────────────────────────────
# List any columns you want to exclude from the model
# (e.g. set-points, calculated values, identifiers)
EXCLUDE_COLS = [
    'Temperature_Setpoint',   # <-- replace with columns to exclude
    # add more columns to exclude here
]

# ══════════════════════════════════════════════════════════════
# SETTINGS BELOW — usually fine to leave as defaults
# ══════════════════════════════════════════════════════════════

# Rolling window sizes (number of time steps for rolling mean/std)
ROLLING_WINDOWS = [10, 30]

# Max samples per fault class during training
# Set to None to use all data (may be slow for very large datasets)
SUBSAMPLE_PER_CLASS = 8000

# 80% train, 20% test
TEST_SIZE = 0.20

# Bootstrap iterations for confidence intervals
N_BOOTSTRAP = 300

# Include Gradient Boosting baseline? (slower but more thorough comparison)
INCLUDE_GRADIENT_BOOSTING = True

# Random seed for reproducibility
RANDOM_STATE = 42
