# pipeline/shap_reliability.py

"""
shap_reliability.py

This module implements SHAP cosine reliability for CRYPTO AI.

Purpose:
- Convert the live SHAP explanation into a numerical vector.
- Build a cluster-level mean SHAP-style vector from saved training artifacts.
- Compare the live SHAP vector with the cluster mean vector using cosine similarity.
- Return a reliability score between 0 and 1.

Why this matters:
The extended project specification requires:

    SHAP prediction reliability via cosine similarity to cluster mean

This module provides that reliability signal and makes it available to:
- Intelligent Score
- ReAct routing
- Streamlit dashboard
- LangSmith tracing

Important:
This module does NOT retrain SVM, DBSCAN, or FCM.
It uses saved training artifacts already loaded in live inference mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ==============================
# General Helpers
# ==============================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to float.
    """

    try:
        number = float(value)
    except Exception:
        number = float(default)

    if np.isnan(number) or np.isinf(number):
        return float(default)

    return number


def normalize_non_negative_vector(vector: np.ndarray) -> np.ndarray:
    """
    Converts a vector into a stable non-negative normalized vector.

    SHAP importance values are expected to be non-negative, but this function
    also protects against missing values, negative values, and zero vectors.
    """

    vector = np.asarray(vector, dtype=float)

    vector = np.nan_to_num(
        vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    vector = np.abs(vector)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_similarity_01(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculates cosine similarity and returns a score between 0 and 1.

    Since the vectors are non-negative, cosine should naturally be in [0, 1].
    Still, we clamp the result for safety.
    """

    vector_a = normalize_non_negative_vector(vector_a)
    vector_b = normalize_non_negative_vector(vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))

    if cosine < 0:
        cosine = 0.0

    if cosine > 1:
        cosine = 1.0

    return round(cosine, 4)


# ==============================
# SHAP Vector Builders
# ==============================

def build_live_shap_vector(
    importance_df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Builds a live SHAP vector aligned with feature_cols.

    Expected importance_df columns:
    - feature
    - shap_importance

    If a feature is missing from importance_df, its importance is set to 0.
    """

    if importance_df is None or importance_df.empty:
        return np.zeros(len(feature_cols), dtype=float)

    if "feature" not in importance_df.columns or "shap_importance" not in importance_df.columns:
        return np.zeros(len(feature_cols), dtype=float)

    importance_map = {}

    for _, row in importance_df.iterrows():
        feature = str(row.get("feature", "")).strip()
        importance = safe_float(row.get("shap_importance", 0.0), default=0.0)

        if feature:
            importance_map[feature] = importance

    vector = np.array(
        [
            importance_map.get(feature, 0.0)
            for feature in feature_cols
        ],
        dtype=float,
    )

    return normalize_non_negative_vector(vector)


def select_cluster_column(training_labeled_df: pd.DataFrame) -> str:
    """
    Selects the best available cluster/context column.

    Priority:
    1. fcm_cluster
    2. dbscan_cluster
    3. risk_label
    4. business_risk_label
    5. svm_predicted_cipher_type
    """

    candidate_columns = [
        "fcm_cluster",
        "dbscan_cluster",
        "risk_label",
        "business_risk_label",
        "svm_predicted_cipher_type",
    ]

    for column in candidate_columns:
        if column in training_labeled_df.columns:
            return column

    return ""


def get_live_cluster_value(
    live_df: pd.DataFrame,
    cluster_column: str,
    predicted_class: str,
) -> Any:
    """
    Gets the live case's cluster/context value.

    If unavailable, fall back to predicted_class.
    """

    if (
        cluster_column
        and live_df is not None
        and not live_df.empty
        and cluster_column in live_df.columns
    ):
        try:
            value = live_df[cluster_column].iloc[0]

            if pd.notna(value):
                return value

        except Exception:
            pass

    return predicted_class


def build_cluster_mean_shap_vector(
    training_labeled_df: pd.DataFrame,
    feature_cols: List[str],
    cluster_column: str,
    cluster_value: Any,
    live_shap_vector: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds a cluster-level SHAP-style mean vector.

    Since saved training artifacts may not contain raw local SHAP vectors for
    every training row, we compute a SHAP-weighted cluster profile:

    1. Select saved training rows belonging to the live case cluster.
    2. Compute the cluster mean feature profile.
    3. Compare it to the global training feature mean.
    4. Weight the cluster profile by the live SHAP importance vector.

    This gives us a stable cluster SHAP-reference vector without retraining.
    """

    if training_labeled_df is None or training_labeled_df.empty:
        return np.zeros(len(feature_cols), dtype=float), {
            "matched_rows": 0,
            "fallback_used": True,
            "reason": "Training dataframe is empty.",
        }

    available_features = [
        feature for feature in feature_cols
        if feature in training_labeled_df.columns
    ]

    if not available_features:
        return np.zeros(len(feature_cols), dtype=float), {
            "matched_rows": 0,
            "fallback_used": True,
            "reason": "No feature columns are available in training dataframe.",
        }

    df = training_labeled_df.copy()

    # Make features numeric.
    for feature in available_features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        median_value = df[feature].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[feature] = df[feature].fillna(median_value)

    fallback_used = False
    reason = ""

    # Select rows from the matching cluster/context.
    if cluster_column and cluster_column in df.columns:
        mask = df[cluster_column].astype(str) == str(cluster_value)
        cluster_df = df.loc[mask].copy()
    else:
        cluster_df = pd.DataFrame()

    # Fallback if cluster has no rows.
    if cluster_df.empty:
        fallback_used = True

        if "svm_predicted_cipher_type" in df.columns:
            mask = df["svm_predicted_cipher_type"].astype(str) == str(cluster_value)
            cluster_df = df.loc[mask].copy()
            reason = "No exact cluster match; used svm_predicted_cipher_type fallback."

        if cluster_df.empty and "risk_label" in df.columns:
            mask = df["risk_label"].astype(str) == str(cluster_value)
            cluster_df = df.loc[mask].copy()
            reason = "No SVM class match; used risk_label fallback."

        if cluster_df.empty:
            cluster_df = df.copy()
            reason = "No matching cluster/class rows; used all training rows fallback."

    global_mean = df[available_features].mean()
    global_std = df[available_features].std().replace(0, 1.0).fillna(1.0)
    cluster_mean = cluster_df[available_features].mean()

    # Cluster profile: how distinctive this cluster is compared with global training data.
    cluster_profile = ((cluster_mean - global_mean).abs() / global_std).fillna(0.0)

    cluster_profile_aligned = np.array(
        [
            safe_float(cluster_profile.get(feature, 0.0), default=0.0)
            if feature in available_features
            else 0.0
            for feature in feature_cols
        ],
        dtype=float,
    )

    # Weight by live SHAP importance.
    live_shap_vector = np.asarray(live_shap_vector, dtype=float)

    if live_shap_vector.shape[0] != len(feature_cols):
        live_shap_vector = np.zeros(len(feature_cols), dtype=float)

    if np.linalg.norm(live_shap_vector) > 0:
        cluster_shap_mean_vector = cluster_profile_aligned * live_shap_vector
    else:
        cluster_shap_mean_vector = cluster_profile_aligned

    cluster_shap_mean_vector = normalize_non_negative_vector(
        cluster_shap_mean_vector
    )

    details = {
        "matched_rows": int(len(cluster_df)),
        "cluster_column": cluster_column,
        "cluster_value": str(cluster_value),
        "fallback_used": fallback_used,
        "reason": reason,
        "available_feature_count": len(available_features),
    }

    return cluster_shap_mean_vector, details


# ==============================
# Main Reliability Function
# ==============================

def calculate_shap_cluster_reliability(
    importance_df: pd.DataFrame,
    feature_cols: List[str],
    training_labeled_df: pd.DataFrame,
    live_df: pd.DataFrame,
    predicted_class: str,
) -> Dict[str, Any]:
    """
    Calculates SHAP cosine reliability against the relevant cluster mean.

    Returns:
    {
        "success": True,
        "method": "shap_cosine_to_cluster_mean",
        "shap_cosine_similarity": 0.0-1.0,
        "cluster_column": "...",
        "cluster_value": "...",
        "matched_rows": int,
        "fallback_used": bool,
        "live_shap_vector": [...],
        "cluster_mean_shap_vector": [...]
    }
    """

    feature_cols = [
        str(feature)
        for feature in feature_cols
    ]

    live_shap_vector = build_live_shap_vector(
        importance_df=importance_df,
        feature_cols=feature_cols,
    )

    cluster_column = select_cluster_column(training_labeled_df)

    cluster_value = get_live_cluster_value(
        live_df=live_df,
        cluster_column=cluster_column,
        predicted_class=predicted_class,
    )

    cluster_mean_vector, cluster_details = build_cluster_mean_shap_vector(
        training_labeled_df=training_labeled_df,
        feature_cols=feature_cols,
        cluster_column=cluster_column,
        cluster_value=cluster_value,
        live_shap_vector=live_shap_vector,
    )

    cosine_score = cosine_similarity_01(
        live_shap_vector,
        cluster_mean_vector,
    )

    reliability_level = "low"

    if cosine_score >= 0.85:
        reliability_level = "high"
    elif cosine_score >= 0.65:
        reliability_level = "medium"

    return {
        "success": True,
        "method": "shap_cosine_to_cluster_mean",
        "description": (
            "SHAP reliability is calculated as cosine similarity between "
            "the live SHAP vector and the matched cluster mean SHAP-style vector."
        ),
        "shap_cosine_similarity": cosine_score,
        "shap_reliability": cosine_score,
        "reliability_level": reliability_level,
        "cluster_column": cluster_column,
        "cluster_value": str(cluster_value),
        "matched_rows": cluster_details.get("matched_rows", 0),
        "fallback_used": cluster_details.get("fallback_used", False),
        "fallback_reason": cluster_details.get("reason", ""),
        "feature_count": len(feature_cols),
        "live_shap_vector": [
            round(float(value), 6)
            for value in live_shap_vector.tolist()
        ],
        "cluster_mean_shap_vector": [
            round(float(value), 6)
            for value in cluster_mean_vector.tolist()
        ],
    }