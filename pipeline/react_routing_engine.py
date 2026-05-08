# pipeline/react_routing_engine.py

"""
react_routing_engine.py

Formal ReAct Routing Engine for CRYPTO AI.

Purpose:
- Convert model outputs into a final adaptive decision route.
- Implement Observe -> Reason -> Act -> Observe.
- Cover all seven required routing paths:
  1. Direct dispatch
  2. SHAP re-check loop
  3. Borderline RAG majority vote
  4. Human review for low confidence
  5. RAG tie handling
  6. RAG zero-success handling
  7. Score below minimum threshold

The action is dynamically computed from live score, SHAP reliability,
RAG outcome, ARIMA/SARIMA trend, SVM confidence, dynamic severity,
dominant SHAP feature, and decoded content.
"""

from __future__ import annotations

from typing import Any, Dict, List


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def classify_trend(forecast_risk_score: float, expected_change_pct: float) -> str:
    forecast_risk_score = clamp01(forecast_risk_score)
    try:
        expected_change_pct = float(expected_change_pct)
    except Exception:
        expected_change_pct = 0.0
    if forecast_risk_score >= 0.70 or expected_change_pct >= 8:
        return "rising"
    if forecast_risk_score <= 0.30 or expected_change_pct <= -8:
        return "declining"
    return "stable"


def route_priority(route_id: str) -> int:
    priorities = {
        "R7_SCORE_BELOW_MINIMUM": 100,
        "R5_RAG_TIE": 95,
        "R6_RAG_ZERO_SUCCESS": 94,
        "R4_HUMAN_REVIEW_LOW_CONFIDENCE": 90,
        "R2_SHAP_RECHECK_LOW_RELIABILITY": 80,
        "R3_BORDERLINE_RAG_MAJORORITY": 75,
        "R1_DIRECT_DISPATCH": 70,
        "R0_MONITOR": 10,
    }
    return priorities.get(route_id, 0)


def calculate_action_intensity(
    total_action_score: float,
    dynamic_severity: float,
    svm_confidence: float,
    shap_cosine_similarity: float,
    forecast_risk_score: float,
) -> str:
    combined = (
        0.30 * clamp01(total_action_score)
        + 0.25 * clamp01(dynamic_severity)
        + 0.20 * clamp01(svm_confidence)
        + 0.15 * clamp01(shap_cosine_similarity)
        + 0.10 * clamp01(forecast_risk_score)
    )
    if combined >= 0.85:
        return "critical"
    if combined >= 0.70:
        return "high"
    if combined >= 0.50:
        return "medium"
    return "low"


def calculate_action_timing(route_id: str, intensity: str, trend_direction: str) -> str:
    if route_id in {"R5_RAG_TIE", "R6_RAG_ZERO_SUCCESS", "R4_HUMAN_REVIEW_LOW_CONFIDENCE"}:
        return "immediate_manual_review"
    if route_id == "R2_SHAP_RECHECK_LOW_RELIABILITY":
        return "immediate_model_recheck"
    if intensity == "critical":
        return "immediate"
    if intensity == "high" and trend_direction == "rising":
        return "immediate"
    if intensity == "high":
        return "within_15_minutes"
    if intensity == "medium":
        return "within_1_hour"
    return "monitor_next_cycle"


def build_personalized_action_message(
    route_id: str,
    predicted_class: str,
    dominant_shap_feature: str,
    decoded_plain_text: str,
    trend_direction: str,
    rag_vote: str,
    intensity: str,
) -> str:
    predicted_class = safe_text(predicted_class) or "Unknown"
    dominant_shap_feature = safe_text(dominant_shap_feature) or "not available"
    decoded_preview = safe_text(decoded_plain_text)[:180] or "not decoded"
    rag_vote = safe_text(rag_vote) or "not_applicable"

    if route_id == "R1_DIRECT_DISPATCH":
        return (
            f"Dispatch an automated SOC alert. The case was classified as {predicted_class} "
            f"with {intensity} intensity. Dominant SHAP evidence: {dominant_shap_feature}. "
            f"ARIMA trend: {trend_direction}. Decoded preview: {decoded_preview}."
        )
    if route_id == "R2_SHAP_RECHECK_LOW_RELIABILITY":
        return (
            f"Do not dispatch immediately. Re-check SHAP reliability because cosine similarity "
            f"is below threshold. Dominant feature: {dominant_shap_feature}. Decoded preview: {decoded_preview}."
        )
    if route_id == "R3_BORDERLINE_RAG_MAJORORITY":
        return (
            f"Use borderline RAG majority decision. RAG voted {rag_vote}; ARIMA trend is "
            f"{trend_direction}; handle with {intensity} intensity."
        )
    if route_id == "R4_HUMAN_REVIEW_LOW_CONFIDENCE":
        return (
            f"Route to human review because model confidence is too low for autonomous dispatch. "
            f"Prediction={predicted_class}; dominant SHAP feature={dominant_shap_feature}."
        )
    if route_id == "R5_RAG_TIE":
        return "Route to human review because RAG evidence produced a tie."
    if route_id == "R6_RAG_ZERO_SUCCESS":
        return "Route to SHAP re-check or analyst review because RAG retrieved no useful voting evidence."
    if route_id == "R7_SCORE_BELOW_MINIMUM":
        return f"Monitor only. Score is below minimum threshold. Prediction={predicted_class}; trend={trend_direction}."
    return f"Monitor the case. Prediction={predicted_class}; trend={trend_direction}."


def evaluate_score_below_minimum(total_action_score: float, minimum_score_threshold: float) -> Dict[str, Any] | None:
    if clamp01(total_action_score) < clamp01(minimum_score_threshold):
        return {
            "route_id": "R7_SCORE_BELOW_MINIMUM",
            "route_name": "Score Below Minimum Threshold",
            "recommended_route": "monitor",
            "action_required": False,
            "reason": "Total action score is below the minimum threshold, so the system monitors only.",
        }
    return None


def evaluate_rag_tie(rag_majority_result: Dict[str, Any]) -> Dict[str, Any] | None:
    if rag_majority_result.get("rag_used") and rag_majority_result.get("tie_detected"):
        return {
            "route_id": "R5_RAG_TIE",
            "route_name": "RAG Tie Handling",
            "recommended_route": "human_review",
            "action_required": False,
            "reason": "RAG evidence produced a tie; route to human review.",
        }
    return None


def evaluate_rag_zero_success(rag_majority_result: Dict[str, Any]) -> Dict[str, Any] | None:
    if rag_majority_result.get("rag_used") and rag_majority_result.get("zero_success"):
        return {
            "route_id": "R6_RAG_ZERO_SUCCESS",
            "route_name": "RAG Zero-Success Handling",
            "recommended_route": rag_majority_result.get("recommended_route", "shap_recheck"),
            "action_required": False,
            "reason": "RAG retrieved no useful evidence; fallback to SHAP re-check or human review.",
        }
    return None


def evaluate_low_confidence_review(svm_confidence: float, low_confidence_threshold: float) -> Dict[str, Any] | None:
    if clamp01(svm_confidence) < clamp01(low_confidence_threshold):
        return {
            "route_id": "R4_HUMAN_REVIEW_LOW_CONFIDENCE",
            "route_name": "Human Review for Low Confidence",
            "recommended_route": "human_review",
            "action_required": False,
            "reason": "SVM confidence is below the low-confidence threshold.",
        }
    return None


def evaluate_shap_recheck(
    shap_cosine_similarity: float,
    shap_reliability_threshold: float,
    total_action_score: float,
    minimum_score_threshold: float,
) -> Dict[str, Any] | None:
    if clamp01(shap_cosine_similarity) < clamp01(shap_reliability_threshold) and clamp01(total_action_score) >= clamp01(minimum_score_threshold):
        return {
            "route_id": "R2_SHAP_RECHECK_LOW_RELIABILITY",
            "route_name": "SHAP Re-check Loop",
            "recommended_route": "shap_recheck",
            "action_required": False,
            "reason": "SHAP cosine reliability is below threshold; re-check explanation reliability before action.",
        }
    return None


def evaluate_borderline_rag(rag_majority_result: Dict[str, Any]) -> Dict[str, Any] | None:
    if rag_majority_result.get("case_type") == "borderline":
        recommended_route = rag_majority_result.get("recommended_route", "human_review")
        return {
            "route_id": "R3_BORDERLINE_RAG_MAJORORITY",
            "route_name": "Borderline RAG Majority Vote",
            "recommended_route": recommended_route,
            "action_required": recommended_route == "dispatch",
            "reason": rag_majority_result.get("reason", "Borderline case handled by explicit RAG majority vote."),
        }
    return None


def evaluate_direct_dispatch(
    total_action_score: float,
    action_threshold: float,
    svm_confidence: float,
    high_confidence_threshold: float,
    shap_cosine_similarity: float,
    shap_reliability_threshold: float,
) -> Dict[str, Any] | None:
    if (
        clamp01(total_action_score) >= clamp01(action_threshold)
        and clamp01(svm_confidence) >= clamp01(high_confidence_threshold)
        and clamp01(shap_cosine_similarity) >= clamp01(shap_reliability_threshold)
    ):
        return {
            "route_id": "R1_DIRECT_DISPATCH",
            "route_name": "Direct Dispatch",
            "recommended_route": "dispatch",
            "action_required": True,
            "reason": "Score, SVM confidence, and SHAP cosine reliability are all high; autonomous dispatch is allowed.",
        }
    return None


def run_react_routing_engine(
    total_action_score: float,
    action_required: bool,
    svm_confidence: float,
    dynamic_severity: float,
    shap_cosine_similarity: float,
    shap_reliability_level: str,
    rag_majority_result: Dict[str, Any],
    forecast_risk_score: float,
    expected_change_pct: float,
    dominant_shap_feature: str,
    predicted_class: str,
    decoded_plain_text: str,
    action_threshold: float = 0.75,
    minimum_score_threshold: float = 0.35,
    low_confidence_threshold: float = 0.45,
    high_confidence_threshold: float = 0.70,
    shap_reliability_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Runs formal ReAct routing and returns the final adaptive route."""

    total_action_score = clamp01(total_action_score)
    svm_confidence = clamp01(svm_confidence)
    dynamic_severity = clamp01(dynamic_severity)
    shap_cosine_similarity = clamp01(shap_cosine_similarity)
    forecast_risk_score = clamp01(forecast_risk_score)

    trend_direction = classify_trend(forecast_risk_score, expected_change_pct)

    observe_state = {
        "total_action_score": round(total_action_score, 4),
        "initial_action_required": bool(action_required),
        "svm_confidence": round(svm_confidence, 4),
        "dynamic_severity": round(dynamic_severity, 4),
        "shap_cosine_similarity": round(shap_cosine_similarity, 4),
        "shap_reliability_level": shap_reliability_level,
        "rag_case_type": rag_majority_result.get("case_type"),
        "rag_used": rag_majority_result.get("rag_used"),
        "rag_majority_vote": rag_majority_result.get("majority_vote"),
        "rag_recommended_route": rag_majority_result.get("recommended_route"),
        "forecast_risk_score": round(forecast_risk_score, 4),
        "expected_change_pct": expected_change_pct,
        "trend_direction": trend_direction,
        "dominant_shap_feature": dominant_shap_feature,
        "predicted_class": predicted_class,
    }

    route_candidates: List[Dict[str, Any]] = []
    for candidate in [
        evaluate_score_below_minimum(total_action_score, minimum_score_threshold),
        evaluate_rag_tie(rag_majority_result),
        evaluate_rag_zero_success(rag_majority_result),
        evaluate_low_confidence_review(svm_confidence, low_confidence_threshold),
        evaluate_shap_recheck(shap_cosine_similarity, shap_reliability_threshold, total_action_score, minimum_score_threshold),
        evaluate_borderline_rag(rag_majority_result),
        evaluate_direct_dispatch(total_action_score, action_threshold, svm_confidence, high_confidence_threshold, shap_cosine_similarity, shap_reliability_threshold),
    ]:
        if candidate:
            candidate["priority"] = route_priority(candidate["route_id"])
            route_candidates.append(candidate)

    if route_candidates:
        selected_route = sorted(route_candidates, key=lambda item: item.get("priority", 0), reverse=True)[0]
    else:
        selected_route = {
            "route_id": "R0_MONITOR",
            "route_name": "Monitor",
            "recommended_route": "monitor",
            "action_required": False,
            "reason": "No high-risk routing condition was met; monitor the case.",
            "priority": route_priority("R0_MONITOR"),
        }

    intensity = calculate_action_intensity(total_action_score, dynamic_severity, svm_confidence, shap_cosine_similarity, forecast_risk_score)
    timing = calculate_action_timing(selected_route["route_id"], intensity, trend_direction)
    rag_vote = rag_majority_result.get("majority_vote", "not_applicable")
    personalized_message = build_personalized_action_message(
        route_id=selected_route["route_id"],
        predicted_class=predicted_class,
        dominant_shap_feature=dominant_shap_feature,
        decoded_plain_text=decoded_plain_text,
        trend_direction=trend_direction,
        rag_vote=rag_vote,
        intensity=intensity,
    )

    adaptive_action = {
        "action_type": selected_route.get("recommended_route", "monitor"),
        "timing": timing,
        "intensity": intensity,
        "personalized_message": personalized_message,
        "computed_from": {
            "total_action_score": round(total_action_score, 4),
            "dynamic_severity": round(dynamic_severity, 4),
            "svm_confidence": round(svm_confidence, 4),
            "shap_cosine_similarity": round(shap_cosine_similarity, 4),
            "rag_vote": rag_vote,
            "arima_trend": trend_direction,
            "dominant_shap_feature": dominant_shap_feature,
        },
    }

    reason_state = {
        "routes_evaluated": route_candidates,
        "selected_route": selected_route,
        "why_selected": selected_route.get("reason"),
        "thresholds": {
            "action_threshold": action_threshold,
            "minimum_score_threshold": minimum_score_threshold,
            "low_confidence_threshold": low_confidence_threshold,
            "high_confidence_threshold": high_confidence_threshold,
            "shap_reliability_threshold": shap_reliability_threshold,
        },
    }

    act_state = {
        "recommended_route": selected_route.get("recommended_route"),
        "action_required": bool(selected_route.get("action_required")),
        "adaptive_action": adaptive_action,
    }

    observe_after_action_state = {
        "status": "pending_runtime_action_result",
        "description": "Actual API/log/email result will be attached after action execution.",
    }

    return {
        "success": True,
        "method": "formal_react_routing_engine",
        "route_id": selected_route["route_id"],
        "route_name": selected_route["route_name"],
        "recommended_route": selected_route["recommended_route"],
        "action_required": bool(selected_route.get("action_required")),
        "reason": selected_route.get("reason"),
        "adaptive_action": adaptive_action,
        "react_trace": {
            "observe": observe_state,
            "reason": reason_state,
            "act": act_state,
            "observe_after_action": observe_after_action_state,
        },
        "routes_evaluated": route_candidates,
        "all_required_paths_supported": [
            "R1_DIRECT_DISPATCH",
            "R2_SHAP_RECHECK_LOW_RELIABILITY",
            "R3_BORDERLINE_RAG_MAJORORITY",
            "R4_HUMAN_REVIEW_LOW_CONFIDENCE",
            "R5_RAG_TIE",
            "R6_RAG_ZERO_SUCCESS",
            "R7_SCORE_BELOW_MINIMUM",
        ],
    }
