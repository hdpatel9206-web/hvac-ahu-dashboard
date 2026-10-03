"""
model_utils.py
--------------
All model inference and SHAP explanation logic.
Keeps prediction, probability extraction, and interpretability
calculations separate from both data loading and visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_MAP, SHAP_SAMPLE_SIZE


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_single(
    model,
    feature_cols: list[str],
    input_values: dict[str, float],
) -> tuple[int, str, dict[str, float]]:
    """
    Run inference on a single manually-entered sensor reading.

    Parameters
    ----------
    model : sklearn estimator
    feature_cols : list[str]
    input_values : dict[str, float]
        {feature_name: value} mapping from the UI sliders.

    Returns
    -------
    (encoded_label, class_name, probabilities_dict)
        probabilities_dict maps class name → probability (0–1).
    """
    input_df = pd.DataFrame([input_values])[feature_cols]

    encoded   = int(model.predict(input_df)[0])
    class_name = LABEL_MAP.get(encoded, "Unknown")
    raw_proba  = model.predict_proba(input_df)[0]

    proba_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(raw_proba)}
    return encoded, class_name, proba_dict


# ── SHAP computations ────────────────────────────────────────────────────────

def compute_shap_values(
    explainer,
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = SHAP_SAMPLE_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray | list, pd.DataFrame]:
    """
    Compute SHAP values for a (sampled) subset of the dataframe.

    For a multi-class Random Forest, ``shap_values`` is a list of
    arrays — one per class — each shaped (n_samples, n_features).

    Parameters
    ----------
    explainer : shap.TreeExplainer
    df : pd.DataFrame
    feature_cols : list[str]
    sample_size : int
    random_state : int

    Returns
    -------
    (shap_values, X_sample)
        shap_values : list[np.ndarray] (one array per class)
        X_sample    : pd.DataFrame of the rows that were sampled
    """
    n = min(len(df), sample_size)
    X_sample = df[feature_cols].sample(n=n, random_state=random_state)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample


def compute_shap_single(
    explainer,
    input_values: dict[str, float],
    feature_cols: list[str],
) -> tuple[np.ndarray | list, pd.DataFrame]:
    """
    Compute SHAP values for a single observation (live prediction).

    Parameters
    ----------
    explainer : shap.TreeExplainer
    input_values : dict[str, float]
    feature_cols : list[str]

    Returns
    -------
    (shap_values, X_single)
    """
    X_single    = pd.DataFrame([input_values])[feature_cols]
    shap_values = explainer.shap_values(X_single)
    return shap_values, X_single


def mean_abs_shap_per_class(
    shap_values: list[np.ndarray],
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature per class.

    Parameters
    ----------
    shap_values : list[np.ndarray]
        Output of ``compute_shap_values``.
    feature_cols : list[str]

    Returns
    -------
    pd.DataFrame
        Index = feature names, columns = class names,
        values = mean absolute SHAP contribution.
    """
    records = {}
    for class_idx, sv in enumerate(shap_values):
        class_name = LABEL_MAP.get(class_idx, f"Class {class_idx}")
        records[class_name] = np.abs(sv).mean(axis=0)

    return pd.DataFrame(records, index=feature_cols)


def overall_feature_importance(
    shap_values: list[np.ndarray],
    feature_cols: list[str],
) -> pd.Series:
    """
    Overall feature importance = mean |SHAP| averaged across all classes.

    Parameters
    ----------
    shap_values : list[np.ndarray]
    feature_cols : list[str]

    Returns
    -------
    pd.Series sorted descending.
    """
    all_sv = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
    return pd.Series(all_sv.mean(axis=0), index=feature_cols).sort_values(ascending=False)


# ── Performance metrics ──────────────────────────────────────────────────────

def compute_per_class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class precision, recall, and F1 from the predictions dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Actual_Label`` and ``Predicted_Label`` columns.

    Returns
    -------
    pd.DataFrame with columns [Class, Precision, Recall, F1, Support]
    """
    from sklearn.metrics import classification_report

    report = classification_report(
        df["Actual_Label"],
        df["Predicted_Label"],
        labels=list(LABEL_MAP.keys()),
        target_names=list(LABEL_MAP.values()),
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for class_name in LABEL_MAP.values():
        r = report.get(class_name, {})
        rows.append({
            "Class":     class_name,
            "Precision": round(r.get("precision", 0), 3),
            "Recall":    round(r.get("recall",    0), 3),
            "F1 Score":  round(r.get("f1-score",  0), 3),
            "Support":   int(r.get("support",     0)),
        })

    return pd.DataFrame(rows)
