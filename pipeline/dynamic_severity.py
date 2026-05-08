# pipeline/dynamic_severity.py

"""
dynamic_severity.py

Dynamic Severity Weighting module for CRYPTO AI.

Purpose:
- Replace fixed class severity S with a case-specific severity value.
- Compute S dynamically from the live case context.
- Produce a full audit trail explaining why the severity score was assigned.

This supports the extended bonus requirement:

    Dynamic severity weighting:
    replacing the fixed class weight S with a value computed per case,
    with justification logged for audit trail.

Inputs:
- SVM predicted class
- Business/risk label
- Decoded plain text
- Dominant SHAP feature
- SVM confidence
- FCM membership
- ARIMA / SARIMA expected trend
- Forecast risk
- Drift score

Output:
{
    "dynamic_severity": 0.0-1.0,
    "severity_level": "LOW/MEDIUM/HIGH/CRITICAL",
    "reason_codes": [...],
    "audit_trail": {...}
}
"""

from __future__ import annotations

from typing import Any, Dict, List


# ==============================
# Helpers
# ==============================

def clamp01(value: Any, default: float = 0.0) -> float:
    """
    Converts any value to float and clamps it between 0 and 1.
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


def normalize_pct(value: Any, default: float = 0.0) -> float:
    """
    Converts a percentage or score to a 0-1 range.
    """

    try:
        number = float(value)
    except Exception:
        number = float(default)

    if abs(number) > 1:
        number = number / 100.0

    return clamp01(abs(number), default=default)


def severity_level(score: float) -> str:
    """
    Converts severity score into a readable level.
    """

    score = clamp01(score)

    if score >= 0.85:
        return "CRITICAL"

    if score >= 0.70:
        return "HIGH"

    if score >= 0.45:
        return "MEDIUM"

    return "LOW"


def get_dominant_shap_feature(top_shap_features: List[Dict[str, Any]] | None) -> str:
    """
    Extracts the dominant SHAP feature from SHAP English node output.
    """

    if not top_shap_features:
        return ""

    first = top_shap_features[0]

    if isinstance(first, dict):
        return str(
            first.get("feature")
            or first.get("readable_feature")
            or ""
        )

    return ""


# ==============================
# Signal Components
# ==============================

def cipher_class_prior(predicted_class: str) -> tuple[float, str]:
    """
    Base severity by predicted cipher/state.

    This is only one component, not the final S.
    """

    predicted = str(predicted_class or "").lower()

    if "anomaly" in predicted:
        return 0.92, "Predicted class is anomaly-like; starting severity is critical."

    if "morse" in predicted:
        return 0.76, "Morse is treated as higher-risk because it can hide commands in symbolic text."

    if "base32" in predicted:
        return 0.64, "Base32 indicates encoded content with moderate-to-high suspicion."

    if "base64" in predicted:
        return 0.56, "Base64 is common but still suspicious in threat-message context."

    return 0.50, "Unknown or unsupported class; using neutral base severity."


def business_label_modifier(business_risk_label: str) -> tuple[float, str]:
    """
    Modifier from business risk label.
    """

    label = str(business_risk_label or "").lower()

    if "critical" in label:
        return 0.16, "Business risk label indicates critical severity."

    if "high" in label:
        return 0.12, "Business risk label indicates high severity."

    if "medium" in label:
        return 0.06, "Business risk label indicates medium severity."

    if "low" in label:
        return -0.05, "Business risk label indicates low severity."

    return 0.0, "No clear business risk label modifier."


def decoded_text_modifier(decoded_plain_text: str) -> tuple[float, List[str]]:
    """
    Adds severity based on decoded command/threat keywords.
    """

    text = str(decoded_plain_text or "").lower()

    if not text:
        return 0.0, ["No decoded text available; no decoded-content modifier applied."]

    critical_terms = [
        "exfiltrate",
        "ransomware",
        "credential",
        "password",
        "token",
        "private key",
        "delete logs",
        "shutdown",
        "disable security",
        "avoid detection",
        "malware",
        "backdoor",
        "payload",
        "breach",
        "steal",
    ]

    high_terms = [
        "attack",
        "command",
        "alert",
        "soc",
        "suspicious",
        "encoded",
        "trigger",
        "hidden",
        "detection",
        "admin",
        "access",
        "system",
    ]

    modifier = 0.0
    reasons = []

    matched_critical = [
        term for term in critical_terms
        if term in text
    ]

    matched_high = [
        term for term in high_terms
        if term in text
    ]

    if matched_critical:
        modifier += 0.18
        reasons.append(
            "Decoded text contains critical threat indicators: "
            + ", ".join(matched_critical[:6])
        )

    if matched_high:
        modifier += 0.10
        reasons.append(
            "Decoded text contains high-risk operational/security terms: "
            + ", ".join(matched_high[:6])
        )

    if not reasons:
        reasons.append("Decoded text did not contain strong threat keywords.")

    return modifier, reasons


def shap_feature_modifier(dominant_shap_feature: str) -> tuple[float, str]:
    """
    Adds severity based on dominant SHAP evidence.
    """

    feature = str(dominant_shap_feature or "").lower()

    if not feature:
        return 0.0, "No dominant SHAP feature available."

    high_risk_feature_terms = [
        "morse",
        "dot",
        "dash",
        "symbol",
        "entropy",
        "equal",
        "base32",
        "base64",
    ]

    if any(term in feature for term in high_risk_feature_terms):
        return 0.07, (
            "Dominant SHAP feature supports suspicious encoding evidence: "
            f"{dominant_shap_feature}"
        )

    return 0.03, (
        "Dominant SHAP feature contributes weak-to-moderate evidence: "
        f"{dominant_shap_feature}"
    )


def model_confidence_modifier(
    svm_confidence: float,
    fcm_membership_score: float,
    shap_cosine_similarity: float | None = None,
) -> tuple[float, List[str]]:
    """
    Adds or reduces severity using model confidence consistency.

    High SVM + high FCM + high SHAP reliability = stronger severity.
    Low consistency = less certainty, may route to review later.
    """

    svm_confidence = clamp01(svm_confidence)
    fcm_membership_score = clamp01(fcm_membership_score, default=0.50)

    if shap_cosine_similarity is None:
        shap_cosine = 0.50
    else:
        shap_cosine = clamp01(shap_cosine_similarity, default=0.50)

    consistency = (
        0.45 * svm_confidence
        + 0.30 * fcm_membership_score
        + 0.25 * shap_cosine
    )

    reasons = [
        f"Model consistency calculated from SVM={svm_confidence:.4f}, "
        f"FCM μ={fcm_membership_score:.4f}, "
        f"SHAP cosine={shap_cosine:.4f}."
    ]

    if consistency >= 0.85:
        return 0.10, reasons + ["High model consistency increases dynamic severity."]

    if consistency >= 0.65:
        return 0.05, reasons + ["Moderate model consistency slightly increases severity."]

    if consistency <= 0.35:
        return -0.08, reasons + ["Low model consistency reduces severity and may require review."]

    return 0.0, reasons + ["Neutral model consistency; no severity modifier applied."]


def arima_trend_modifier(
    forecast_risk_score: float,
    expected_change_pct: float,
    drift_score: float,
) -> tuple[float, List[str]]:
    """
    Adds severity from forecast and trend behavior.
    """

    forecast_risk_score = clamp01(forecast_risk_score)
    drift_score = clamp01(drift_score)

    try:
        expected_change_pct = float(expected_change_pct)
    except Exception:
        expected_change_pct = 0.0

    reasons = [
        f"Forecast risk={forecast_risk_score:.4f}, "
        f"expected change={expected_change_pct:.4f}%, "
        f"drift score={drift_score:.4f}."
    ]

    modifier = 0.0

    if forecast_risk_score >= 0.75:
        modifier += 0.08
        reasons.append("High forecast risk increases severity.")

    elif forecast_risk_score >= 0.55:
        modifier += 0.04
        reasons.append("Moderate forecast risk slightly increases severity.")

    if expected_change_pct >= 10:
        modifier += 0.06
        reasons.append("Rising ARIMA/SARIMA trend increases severity.")

    elif expected_change_pct <= -10:
        modifier -= 0.04
        reasons.append("Declining ARIMA/SARIMA trend dampens severity.")

    if drift_score >= 0.60:
        modifier += 0.05
        reasons.append("High drift score increases severity.")

    return modifier, reasons


# ==============================
# Main Function
# ==============================

def calculate_dynamic_severity(
    predicted_class: str,
    business_risk_label: str = "",
    decoded_plain_text: str = "",
    dominant_shap_feature: str = "",
    svm_confidence: float = 0.0,
    fcm_membership_score: float = 0.50,
    shap_cosine_similarity: float | None = None,
    forecast_risk_score: float = 0.0,
    expected_change_pct: float = 0.0,
    drift_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculates case-specific dynamic class severity S.

    Returns a full audit trail explaining how S was computed.
    """

    reason_codes = []

    base_score, base_reason = cipher_class_prior(predicted_class)
    reason_codes.append(base_reason)

    business_mod, business_reason = business_label_modifier(business_risk_label)
    reason_codes.append(business_reason)

    decoded_mod, decoded_reasons = decoded_text_modifier(decoded_plain_text)
    reason_codes.extend(decoded_reasons)

    shap_mod, shap_reason = shap_feature_modifier(dominant_shap_feature)
    reason_codes.append(shap_reason)

    consistency_mod, consistency_reasons = model_confidence_modifier(
        svm_confidence=svm_confidence,
        fcm_membership_score=fcm_membership_score,
        shap_cosine_similarity=shap_cosine_similarity,
    )
    reason_codes.extend(consistency_reasons)

    arima_mod, arima_reasons = arima_trend_modifier(
        forecast_risk_score=forecast_risk_score,
        expected_change_pct=expected_change_pct,
        drift_score=drift_score,
    )
    reason_codes.extend(arima_reasons)

    raw_score = (
        base_score
        + business_mod
        + decoded_mod
        + shap_mod
        + consistency_mod
        + arima_mod
    )

    final_score = round(clamp01(raw_score, default=0.50), 4)
    level = severity_level(final_score)

    return {
        "success": True,
        "method": "dynamic_case_severity_weighting",
        "dynamic_severity": final_score,
        "severity_level": level,
        "reason_codes": reason_codes,
        "audit_trail": {
            "predicted_class": predicted_class,
            "business_risk_label": business_risk_label,
            "decoded_plain_text_preview": str(decoded_plain_text or "")[:250],
            "dominant_shap_feature": dominant_shap_feature,
            "svm_confidence": round(clamp01(svm_confidence), 4),
            "fcm_membership_score": round(clamp01(fcm_membership_score, default=0.50), 4),
            "shap_cosine_similarity": (
                None
                if shap_cosine_similarity is None
                else round(clamp01(shap_cosine_similarity), 4)
            ),
            "forecast_risk_score": round(clamp01(forecast_risk_score), 4),
            "expected_change_pct": expected_change_pct,
            "drift_score": round(clamp01(drift_score), 4),
            "base_score": round(base_score, 4),
            "business_modifier": round(business_mod, 4),
            "decoded_text_modifier": round(decoded_mod, 4),
            "shap_feature_modifier": round(shap_mod, 4),
            "model_consistency_modifier": round(consistency_mod, 4),
            "arima_trend_modifier": round(arima_mod, 4),
            "raw_score_before_clamp": round(raw_score, 4),
            "final_dynamic_severity": final_score,
        },
    }