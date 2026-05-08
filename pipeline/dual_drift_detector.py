# pipeline/dual_drift_detector.py

"""
dual_drift_detector.py

Dual Drift Detection module for CRYPTO AI.

Purpose:
- Replace single-signal drift triggering with the strict two-signal rule.
- Signal 1: Gap(SVM) / classifier-confidence trend decline.
- Signal 2: Centroid movement in feature space.
- Dynamic reclustering is allowed only when BOTH signals fire together.

This supports the extended requirement:

    Two independent drift signals must both fire simultaneously before
    re-clustering triggers. Signal 1 is the Gap(SVM) trend decline.
    Signal 2 is centroid movement in feature space. Either alone is noise.

Important:
- This module does not retrain anything.
- It only decides whether the outer loop should be triggered.
- The caller decides whether to run reclustering/retraining.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================

def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)

    if np.isnan(number) or np.isinf(number):
        number = float(default)

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)

    if np.isnan(number) or np.isinf(number):
        return float(default)

    return float(number)


def get_first_value(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    try:
        if df is not None and not df.empty and column in df.columns:
            value = df[column].iloc[0]
            if pd.notna(value):
                return value
    except Exception:
        pass

    return default


def pick_cluster_column(training_labeled_df: pd.DataFrame, live_df: pd.DataFrame) -> str:
    """
    Chooses the most useful cluster/context column available.
    """

    candidates = [
        "fcm_cluster",
        "dbscan_cluster",
        "risk_label",
        "business_risk_label",
        "svm_predicted_cipher_type",
    ]

    for column in candidates:
        if (
            training_labeled_df is not None
            and column in training_labeled_df.columns
            and live_df is not None
            and column in live_df.columns
        ):
            return column

    for column in candidates:
        if training_labeled_df is not None and column in training_labeled_df.columns:
            return column

    return ""


# =========================================================
# Signal 1: Gap(SVM) trend decline
# =========================================================

def detect_gap_svm_trend_decline(
    current_svm_confidence: float,
    training_labeled_df: pd.DataFrame,
    confidence_column: str = "svm_confidence",
    absolute_floor: float = 0.45,
    relative_drop_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Detects classifier-confidence degradation.

    The current live confidence is compared against historical confidence from
    the saved training/labeled dataframe.

    A decline is detected if either:
    - current confidence is below an absolute safety floor
    - current confidence drops materially below historical average
    """

    current = clamp01(current_svm_confidence)

    if training_labeled_df is None or training_labeled_df.empty:
        return {
            "signal_name": "gap_svm_trend_decline",
            "detected": False,
            "reason": "Training dataframe is empty; cannot establish historical confidence baseline.",
            "current_svm_confidence": round(current, 4),
            "historical_mean_confidence": None,
            "relative_drop": 0.0,
            "absolute_floor": absolute_floor,
        }

    if confidence_column not in training_labeled_df.columns:
        return {
            "signal_name": "gap_svm_trend_decline",
            "detected": False,
            "reason": f"Column '{confidence_column}' is unavailable; treating signal as not fired.",
            "current_svm_confidence": round(current, 4),
            "historical_mean_confidence": None,
            "relative_drop": 0.0,
            "absolute_floor": absolute_floor,
        }

    historical = pd.to_numeric(
        training_labeled_df[confidence_column],
        errors="coerce",
    ).dropna()

    historical = historical[(historical >= 0) & (historical <= 1)]

    if historical.empty:
        return {
            "signal_name": "gap_svm_trend_decline",
            "detected": False,
            "reason": "Historical confidence values are empty after cleaning.",
            "current_svm_confidence": round(current, 4),
            "historical_mean_confidence": None,
            "relative_drop": 0.0,
            "absolute_floor": absolute_floor,
        }

    historical_mean = float(historical.mean())
    historical_p25 = float(historical.quantile(0.25))

    if historical_mean <= 0:
        relative_drop = 0.0
    else:
        relative_drop = max(0.0, (historical_mean - current) / historical_mean)

    below_floor = current < absolute_floor
    below_historical_quartile = current < historical_p25
    material_drop = relative_drop >= relative_drop_threshold

    detected = bool(below_floor or (below_historical_quartile and material_drop))

    if detected:
        reason = (
            "Gap(SVM) decline detected: current live confidence is materially below "
            "the historical training confidence baseline."
        )
    else:
        reason = (
            "Gap(SVM) decline not detected: current live confidence is within the "
            "acceptable historical range."
        )

    return {
        "signal_name": "gap_svm_trend_decline",
        "detected": detected,
        "reason": reason,
        "current_svm_confidence": round(current, 4),
        "historical_mean_confidence": round(historical_mean, 4),
        "historical_p25_confidence": round(historical_p25, 4),
        "relative_drop": round(float(relative_drop), 4),
        "absolute_floor": round(float(absolute_floor), 4),
        "relative_drop_threshold": round(float(relative_drop_threshold), 4),
        "below_floor": bool(below_floor),
        "below_historical_quartile": bool(below_historical_quartile),
        "material_drop": bool(material_drop),
    }


# =========================================================
# Signal 2: Centroid movement in feature space
# =========================================================

def detect_centroid_movement(
    live_scaled: np.ndarray,
    training_scaled: np.ndarray,
    training_labeled_df: pd.DataFrame,
    live_df: pd.DataFrame,
    cluster_column: str | None = None,
    z_threshold: float = 2.50,
    percentile_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Detects feature-space drift by comparing the live point to the matched
    cluster centroid.

    Detection rule:
    - Select the live case cluster/context.
    - Compute centroid of historical points in the same cluster.
    - Compute live distance to centroid.
    - Compare to historical intra-cluster distance distribution.
    """

    try:
        live_scaled = np.asarray(live_scaled, dtype=float)
        training_scaled = np.asarray(training_scaled, dtype=float)

        if live_scaled.ndim == 2:
            live_vector = live_scaled[0]
        else:
            live_vector = live_scaled.reshape(-1)

        if training_scaled.ndim != 2 or training_scaled.shape[0] == 0:
            raise ValueError("training_scaled must be a non-empty 2D array")

        if training_labeled_df is None or training_labeled_df.empty:
            raise ValueError("training_labeled_df is empty")

        if cluster_column is None:
            cluster_column = pick_cluster_column(training_labeled_df, live_df)

        cluster_value = get_first_value(live_df, cluster_column, default=None) if cluster_column else None

        if cluster_column and cluster_column in training_labeled_df.columns and cluster_value is not None:
            mask = training_labeled_df[cluster_column].astype(str) == str(cluster_value)
            cluster_indices = np.where(mask.to_numpy())[0]
        else:
            cluster_indices = np.array([], dtype=int)

        fallback_used = False
        fallback_reason = ""

        if cluster_indices.size < 5:
            fallback_used = True
            fallback_reason = (
                "Matched cluster had fewer than 5 rows or no cluster column was available; "
                "used all training rows as baseline."
            )
            cluster_indices = np.arange(training_scaled.shape[0])

        cluster_scaled = training_scaled[cluster_indices]
        centroid = cluster_scaled.mean(axis=0)

        historical_distances = np.linalg.norm(cluster_scaled - centroid, axis=1)
        live_distance = float(np.linalg.norm(live_vector - centroid))

        distance_mean = float(np.mean(historical_distances))
        distance_std = float(np.std(historical_distances))

        if distance_std <= 1e-12:
            z_score = 0.0 if live_distance <= distance_mean else 999.0
        else:
            z_score = (live_distance - distance_mean) / distance_std

        percentile_cutoff = float(np.quantile(historical_distances, percentile_threshold))

        above_percentile = live_distance > percentile_cutoff
        above_z = z_score >= z_threshold

        detected = bool(above_percentile and above_z)

        if detected:
            reason = (
                "Centroid movement detected: the live feature vector is outside the "
                "normal intra-cluster distance distribution."
            )
        else:
            reason = (
                "Centroid movement not detected: the live feature vector remains within "
                "the matched cluster's historical feature-space boundary."
            )

        return {
            "signal_name": "centroid_movement_feature_space",
            "detected": detected,
            "reason": reason,
            "cluster_column": cluster_column or "none",
            "cluster_value": str(cluster_value),
            "matched_rows": int(len(cluster_indices)),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "live_distance_to_centroid": round(live_distance, 4),
            "historical_distance_mean": round(distance_mean, 4),
            "historical_distance_std": round(distance_std, 4),
            "historical_distance_percentile_cutoff": round(percentile_cutoff, 4),
            "percentile_threshold": round(float(percentile_threshold), 4),
            "z_score": round(float(z_score), 4),
            "z_threshold": round(float(z_threshold), 4),
            "above_percentile": bool(above_percentile),
            "above_z_threshold": bool(above_z),
        }

    except Exception as e:
        return {
            "signal_name": "centroid_movement_feature_space",
            "detected": False,
            "reason": "Centroid movement could not be evaluated; treating signal as not fired.",
            "error": str(e),
            "cluster_column": cluster_column or "unknown",
            "cluster_value": "unknown",
            "matched_rows": 0,
            "fallback_used": True,
        }


# =========================================================
# Main Dual Drift Detector
# =========================================================

def detect_dual_drift(
    current_svm_confidence: float,
    live_scaled: np.ndarray,
    training_scaled: np.ndarray,
    training_labeled_df: pd.DataFrame,
    live_df: pd.DataFrame,
    existing_forecast_drift_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Runs the strict two-signal drift rule.

    Dynamic reclustering / outer-loop rebuild should trigger only if:
        Gap(SVM) trend decline == True
        AND
        Centroid movement == True
    """

    gap_signal = detect_gap_svm_trend_decline(
        current_svm_confidence=current_svm_confidence,
        training_labeled_df=training_labeled_df,
    )

    centroid_signal = detect_centroid_movement(
        live_scaled=live_scaled,
        training_scaled=training_scaled,
        training_labeled_df=training_labeled_df,
        live_df=live_df,
    )

    gap_detected = bool(gap_signal.get("detected"))
    centroid_detected = bool(centroid_signal.get("detected"))
    dual_drift_detected = bool(gap_detected and centroid_detected)

    if dual_drift_detected:
        recommendation = (
            "Dual drift detected. Both Gap(SVM) trend decline and centroid movement "
            "fired simultaneously, so the outer ReAct rebuild loop is allowed."
        )
        outer_loop_required = True
        decision = "TRIGGER_OUTER_LOOP"
    elif gap_detected or centroid_detected:
        recommendation = (
            "Only one drift signal fired. Per the strict rule, this is treated as noise; "
            "do not recluster or retrain yet."
        )
        outer_loop_required = False
        decision = "NOISE_SINGLE_SIGNAL_ONLY"
    else:
        recommendation = (
            "No dual drift detected. Neither signal combination justifies reclustering."
        )
        outer_loop_required = False
        decision = "NO_DRIFT"

    existing_forecast_drift_result = existing_forecast_drift_result or {}

    return {
        "success": True,
        "method": "dual_drift_detector",
        "dual_drift_detected": dual_drift_detected,
        "outer_loop_required": outer_loop_required,
        "decision": decision,
        "recommendation": recommendation,
        "strict_rule": (
            "Trigger reclustering only when BOTH independent signals fire: "
            "Gap(SVM) trend decline AND centroid movement in feature space."
        ),
        "signals": {
            "gap_svm_trend_decline": gap_signal,
            "centroid_movement": centroid_signal,
        },
        "single_signal_noise": bool((gap_detected or centroid_detected) and not dual_drift_detected),
        "forecast_drift_context": {
            "legacy_drift_detected": bool(existing_forecast_drift_result.get("drift_detected", False)),
            "legacy_drift_score": existing_forecast_drift_result.get("drift_score"),
            "legacy_recommendation": existing_forecast_drift_result.get("recommendation"),
        },
    }
