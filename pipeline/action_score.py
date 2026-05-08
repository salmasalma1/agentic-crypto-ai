# pipeline/action_score.py

"""
action_score.py

Exact Intelligent Score module for CRYPTO AI.

This module calculates the composite intelligent score required by the
extended project specification.

Required independent signals:
1. S              = Class severity
2. Gap(SVM)       = SVM confidence / classifier separation signal
3. μ(predicted)   = FCM membership for the predicted cluster/class context
4. ARIMA          = Forecast / cluster trend multiplier
5. SHAP           = Prediction reliability signal

Important:
- These five signals are kept separate.
- The final score is computed from all five.
- The output includes a full score breakdown for Streamlit, logs, LangSmith,
  and team explanation.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from utils.config import ACTION_THRESHOLD
except Exception:
    ACTION_THRESHOLD = 0.75


# =========================================================
# Normalization Helpers
# =========================================================

def clamp01(value: Any, default: float = 0.0) -> float:
    """
    Converts a value to float and clamps it between 0 and 1.
    """

    try:
        number = float(value)
    except Exception:
        number = float(default)

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def normalize_percentage_or_score(value: Any, default: float = 0.0) -> float:
    """
    Accepts either:
    - 0 to 1 score
    - 0 to 100 percentage

    Returns:
    - 0 to 1 normalized score
    """

    try:
        number = float(value)
    except Exception:
        number = float(default)

    if number > 1:
        number = number / 100.0

    return clamp01(number, default=default)


# =========================================================
# Signal Builders
# =========================================================

def calculate_class_severity(
    predicted_class: str = "",
    business_risk_label: str = "",
    decoded_plain_text: str = "",
    base_weight: float | None = None,
) -> float:
    """
    Calculates S = class severity.

    This is not a fixed lookup only. It combines:
    - predicted cipher/state
    - business risk label
    - decoded text hints
    - optional base weight

    The result is normalized between 0 and 1.
    """

    predicted_class = str(predicted_class or "").strip().lower()
    business_risk_label = str(business_risk_label or "").strip().lower()
    decoded_plain_text = str(decoded_plain_text or "").strip().lower()

    # Start from supplied base weight if available.
    if base_weight is not None:
        severity = clamp01(base_weight, default=0.50)
    else:
        severity = 0.50

    # Cipher/state prior.
    if "anomaly" in predicted_class:
        severity = max(severity, 0.90)
    elif "morse" in predicted_class:
        severity = max(severity, 0.78)
    elif "base32" in predicted_class:
        severity = max(severity, 0.66)
    elif "base64" in predicted_class:
        severity = max(severity, 0.58)

    # Business label prior.
    if "critical" in business_risk_label:
        severity = max(severity, 0.95)
    elif "high" in business_risk_label:
        severity = max(severity, 0.85)
    elif "medium" in business_risk_label:
        severity = max(severity, 0.65)
    elif "low" in business_risk_label:
        severity = max(severity, 0.35)

    # Decoded command/content hints.
    high_risk_terms = [
        "attack",
        "malware",
        "exfiltrate",
        "breach",
        "steal",
        "delete",
        "shutdown",
        "ransom",
        "credential",
        "password",
        "alert",
        "soc",
        "command",
        "avoid detection",
    ]

    if any(term in decoded_plain_text for term in high_risk_terms):
        severity = max(severity, 0.82)

    return round(clamp01(severity, default=0.50), 4)


def calculate_svm_gap_signal(
    svm_confidence: float,
    second_best_confidence: float | None = None,
) -> float:
    """
    Calculates Gap(SVM).

    If second_best_confidence is available:
        Gap = top confidence - second confidence

    If not available:
        fallback = top confidence

    This keeps compatibility with the current SVM output where only the top
    probability/confidence may be available.
    """

    top = clamp01(svm_confidence, default=0.0)

    if second_best_confidence is None:
        return round(top, 4)

    second = clamp01(second_best_confidence, default=0.0)
    gap = max(0.0, top - second)

    return round(clamp01(gap, default=0.0), 4)


def calculate_fcm_membership_signal(
    fcm_membership_score: float | None,
    fallback: float = 0.50,
) -> float:
    """
    Calculates μ(predicted).

    It uses the FCM membership score assigned to the live case.
    """

    if fcm_membership_score is None:
        return round(clamp01(fallback, default=0.50), 4)

    return round(clamp01(fcm_membership_score, default=fallback), 4)


def calculate_arima_trend_signal(
    forecast_risk_score: float = 0.0,
    expected_change_pct: float = 0.0,
) -> float:
    """
    Calculates ARIMA / SARIMA trend multiplier.

    Inputs:
    - forecast_risk_score: normalized forecast risk
    - expected_change_pct: future trend percentage

    If the trend is rising, the signal becomes stronger.
    If the trend is declining, the signal is dampened.
    """

    base = clamp01(forecast_risk_score, default=0.0)

    try:
        change = float(expected_change_pct)
    except Exception:
        change = 0.0

    # Convert expected percentage change into a small multiplier.
    # Rising trend increases trust/risk; declining trend dampens it.
    if change >= 20:
        multiplier = 1.20
    elif change >= 10:
        multiplier = 1.10
    elif change >= 3:
        multiplier = 1.05
    elif change <= -20:
        multiplier = 0.70
    elif change <= -10:
        multiplier = 0.80
    elif change <= -3:
        multiplier = 0.90
    else:
        multiplier = 1.00

    arima_signal = base * multiplier

    return round(clamp01(arima_signal, default=0.0), 4)


def calculate_shap_reliability_signal(
    shap_importance_score: float = 0.0,
    shap_cosine_similarity: float | None = None,
) -> float:
    """
    Calculates SHAP reliability.

    Current supported inputs:
    - SHAP importance score
    - optional SHAP cosine similarity to cluster mean

    Once Phase 2 is added, shap_cosine_similarity will become the preferred
    reliability signal.
    """

    if shap_cosine_similarity is not None:
        return round(clamp01(shap_cosine_similarity, default=0.0), 4)

    return round(clamp01(shap_importance_score, default=0.0), 4)


# =========================================================
# Main Intelligent Score
# =========================================================

def calculate_intelligent_score(
    class_severity: float,
    svm_gap: float,
    fcm_membership: float,
    arima_signal: float,
    shap_reliability: float,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Calculates the exact five-signal intelligent score.

    Weights are explicit and auditable.
    """

    threshold = ACTION_THRESHOLD if threshold is None else float(threshold)

    S = clamp01(class_severity)
    gap_svm = clamp01(svm_gap)
    mu_predicted = clamp01(fcm_membership)
    arima = clamp01(arima_signal)
    shap = clamp01(shap_reliability)

    # Balanced composite score.
    # S and SVM confidence carry strong decision value.
    # FCM membership verifies geometric consistency.
    # ARIMA adds trend awareness.
    # SHAP adds reliability/explanation strength.
    weights = {
        "S_class_severity": 0.22,
        "Gap_SVM": 0.23,
        "Mu_predicted_FCM": 0.18,
        "ARIMA_trend": 0.17,
        "SHAP_reliability": 0.20,
    }

    total_score = (
        weights["S_class_severity"] * S
        + weights["Gap_SVM"] * gap_svm
        + weights["Mu_predicted_FCM"] * mu_predicted
        + weights["ARIMA_trend"] * arima
        + weights["SHAP_reliability"] * shap
    )

    action_required = total_score >= threshold

    return {
        "success": True,
        "score_type": "exact_five_signal_intelligent_score",
        "formula": (
            "Intelligent Score = "
            "0.22*S + 0.23*Gap(SVM) + 0.18*μ(predicted) "
            "+ 0.17*ARIMA + 0.20*SHAP"
        ),
        "signals": {
            "S_class_severity": round(S, 4),
            "Gap_SVM": round(gap_svm, 4),
            "Mu_predicted_FCM": round(mu_predicted, 4),
            "ARIMA_trend": round(arima, 4),
            "SHAP_reliability": round(shap, 4),
        },
        "weights": weights,
        "total_action_score": round(float(total_score), 4),
        "threshold": round(float(threshold), 4),
        "action_required": bool(action_required),
        "decision": "ACTION_REQUIRED" if action_required else "MONITOR_ONLY",
    }


# =========================================================
# Backward-Compatible Wrapper
# =========================================================

def calculate_total_action_score(
    svm_confidence: float,
    shap_importance_score: float,
    forecast_risk_score: float,
    drift_score: float = 0.0,
    cipher_risk_weight: float = 0.50,
    fcm_membership_score: float | None = None,
    predicted_class: str = "",
    business_risk_label: str = "",
    decoded_plain_text: str = "",
    expected_change_pct: float = 0.0,
    shap_cosine_similarity: float | None = None,
    second_best_confidence: float | None = None,
) -> Dict[str, Any]:
    """
    Backward-compatible function used by the existing pipeline.

    Older pipeline versions called:
        calculate_total_action_score(svm_confidence, shap_score, forecast_score)

    New pipeline versions can pass:
        fcm_membership_score
        predicted_class
        business_risk_label
        decoded_plain_text
        expected_change_pct
        shap_cosine_similarity

    Returns a dictionary with:
    - xai_score
    - total_action_score
    - action_required
    - five independent signals
    """

    svm_confidence = clamp01(svm_confidence)
    shap_importance_score = clamp01(shap_importance_score)
    forecast_risk_score = clamp01(forecast_risk_score)
    drift_score = clamp01(drift_score)
    cipher_risk_weight = clamp01(cipher_risk_weight, default=0.50)

    xai_score = round(svm_confidence * shap_importance_score, 4)

    class_severity = calculate_class_severity(
        predicted_class=predicted_class,
        business_risk_label=business_risk_label,
        decoded_plain_text=decoded_plain_text,
        base_weight=cipher_risk_weight,
    )

    svm_gap = calculate_svm_gap_signal(
        svm_confidence=svm_confidence,
        second_best_confidence=second_best_confidence,
    )

    mu_predicted = calculate_fcm_membership_signal(
        fcm_membership_score=fcm_membership_score,
        fallback=0.50,
    )

    arima_signal = calculate_arima_trend_signal(
        forecast_risk_score=forecast_risk_score,
        expected_change_pct=expected_change_pct,
    )

    # Drift boosts ARIMA signal only slightly, but keeps it independent.
    if drift_score > 0:
        arima_signal = round(clamp01((arima_signal + drift_score) / 2.0), 4)

    shap_reliability = calculate_shap_reliability_signal(
        shap_importance_score=shap_importance_score,
        shap_cosine_similarity=shap_cosine_similarity,
    )

    intelligent_result = calculate_intelligent_score(
        class_severity=class_severity,
        svm_gap=svm_gap,
        fcm_membership=mu_predicted,
        arima_signal=arima_signal,
        shap_reliability=shap_reliability,
    )

    intelligent_result.update({
        "xai_score": xai_score,
        "xai_formula": "XAI Score = SVM Confidence × SHAP Importance",
        "legacy_inputs": {
            "svm_confidence": round(svm_confidence, 4),
            "shap_importance_score": round(shap_importance_score, 4),
            "forecast_risk_score": round(forecast_risk_score, 4),
            "drift_score": round(drift_score, 4),
            "cipher_risk_weight": round(cipher_risk_weight, 4),
        },
        "score_components": {
            "svm_confidence": round(svm_confidence, 4),
            "shap_importance_score": round(shap_importance_score, 4),
            "xai_score": xai_score,
            "forecast_risk_score": round(forecast_risk_score, 4),
            "drift_score": round(drift_score, 4),
            "cipher_risk_weight": round(cipher_risk_weight, 4),
            "fcm_membership_score": round(mu_predicted, 4),
            "class_severity": round(class_severity, 4),
            "svm_gap": round(svm_gap, 4),
            "arima_signal": round(arima_signal, 4),
            "shap_reliability": round(shap_reliability, 4),
        },
    })

    return intelligent_result