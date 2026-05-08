# pipeline/outcome_tracker.py

"""
outcome_tracker.py

Outcome tracking and lightweight adaptation module for CRYPTO AI.

Purpose:
- Record every live case after the action layer runs.
- Observe whether the action succeeded, failed, or was routed to review/monitoring.
- Compute lightweight between-cycle adaptation guidance from recent outcomes.

This supports the agentic property:
    "adapts to outcomes"

It does not retrain models. It creates a feedback memory that later pipeline
runs can use to adjust review bias, action intensity, and monitoring urgency
between full retraining cycles.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# =========================================================
# Paths
# =========================================================

DEFAULT_OUTCOME_LOG_PATH = Path("artifacts") / "outcome_tracking_log.csv"


# =========================================================
# Helpers
# =========================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)

    if pd.isna(number):
        return float(default)

    return number


def clamp01(value: Any, default: float = 0.0) -> float:
    number = safe_float(value, default=default)

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_case_id(
    cypher_text: str,
    predicted_class: str,
    timestamp: str,
) -> str:
    raw = f"{cypher_text}|{predicted_class}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def json_dumps_safe(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def infer_outcome_status(
    action_required: bool,
    api_result: Dict[str, Any],
    react_route: str,
    recommended_route: str,
) -> str:
    """
    Creates a normalized outcome label from the action result.
    """

    react_route = str(react_route or "")
    recommended_route = str(recommended_route or "")

    if recommended_route in {"human_review", "manual_review"} or "HUMAN_REVIEW" in react_route:
        return "human_review_required"

    if recommended_route == "shap_recheck" or "SHAP_RECHECK" in react_route:
        return "model_recheck_required"

    if not action_required:
        return "monitor_only"

    if api_result.get("success"):
        return "action_success"

    if api_result.get("email_attempted") or api_result.get("action_logged"):
        return "action_partial"

    return "action_failed"


def load_outcome_history(log_path: str | Path = DEFAULT_OUTCOME_LOG_PATH) -> pd.DataFrame:
    """
    Loads previous outcome history if available.
    """

    path = Path(log_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_outcome_row(
    row: Dict[str, Any],
    log_path: str | Path = DEFAULT_OUTCOME_LOG_PATH,
) -> None:
    """
    Appends a single outcome row to CSV.
    """

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])

    if path.exists():
        row_df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")
    else:
        row_df.to_csv(path, index=False, encoding="utf-8")


# =========================================================
# Adaptation Logic
# =========================================================

def summarize_recent_outcomes(
    history_df: pd.DataFrame,
    predicted_class: str,
    react_route: str,
    window_size: int = 25,
) -> Dict[str, Any]:
    """
    Summarizes recent outcomes for the current class and route.
    """

    if history_df is None or history_df.empty:
        return {
            "history_available": False,
            "matched_count": 0,
            "success_rate": None,
            "failure_rate": None,
            "review_rate": None,
            "monitor_rate": None,
        }

    df = history_df.copy()

    # Match both same predicted class and same route when possible.
    if "predicted_class" in df.columns:
        df = df[df["predicted_class"].astype(str) == str(predicted_class)]

    if "react_route" in df.columns and not df.empty:
        same_route = df[df["react_route"].astype(str) == str(react_route)]
        if not same_route.empty:
            df = same_route

    if df.empty:
        return {
            "history_available": True,
            "matched_count": 0,
            "success_rate": None,
            "failure_rate": None,
            "review_rate": None,
            "monitor_rate": None,
        }

    df = df.tail(window_size)

    statuses = df.get("outcome_status", pd.Series(dtype=str)).astype(str).str.lower()
    total = len(statuses)

    if total == 0:
        return {
            "history_available": True,
            "matched_count": 0,
            "success_rate": None,
            "failure_rate": None,
            "review_rate": None,
            "monitor_rate": None,
        }

    success_count = int(statuses.isin(["action_success", "action_partial"]).sum())
    failure_count = int(statuses.isin(["action_failed"]).sum())
    review_count = int(statuses.isin(["human_review_required", "model_recheck_required"]).sum())
    monitor_count = int(statuses.isin(["monitor_only"]).sum())

    return {
        "history_available": True,
        "matched_count": total,
        "success_rate": round(success_count / total, 4),
        "failure_rate": round(failure_count / total, 4),
        "review_rate": round(review_count / total, 4),
        "monitor_rate": round(monitor_count / total, 4),
        "window_size": window_size,
    }


def compute_between_cycle_adaptation(
    recent_summary: Dict[str, Any],
    total_action_score: float,
    dynamic_severity: float,
    current_intensity: str,
) -> Dict[str, Any]:
    """
    Produces lightweight adaptation guidance from recent outcomes.

    This does not modify the saved model. It adjusts operational behavior between
    full retraining cycles.
    """

    total_action_score = clamp01(total_action_score)
    dynamic_severity = clamp01(dynamic_severity)
    failure_rate = recent_summary.get("failure_rate")
    success_rate = recent_summary.get("success_rate")
    review_rate = recent_summary.get("review_rate")
    matched_count = int(recent_summary.get("matched_count") or 0)

    recommendation = "no_history_yet"
    action_bias = "keep_current_policy"
    intensity_adjustment = "none"
    confidence_adjustment = 0.0
    reason_codes: List[str] = []

    if matched_count == 0:
        reason_codes.append("No matched outcome history is available yet for this class/route.")
        return {
            "adaptation_available": False,
            "recommendation": recommendation,
            "action_bias": action_bias,
            "intensity_adjustment": intensity_adjustment,
            "confidence_adjustment": confidence_adjustment,
            "reason_codes": reason_codes,
        }

    failure_rate = safe_float(failure_rate, 0.0)
    success_rate = safe_float(success_rate, 0.0)
    review_rate = safe_float(review_rate, 0.0)

    if failure_rate >= 0.35:
        recommendation = "increase_safety_bias"
        action_bias = "prefer_human_review_for_similar_cases"
        intensity_adjustment = "increase_review_urgency"
        confidence_adjustment = -0.08
        reason_codes.append(
            f"Recent failure rate is high ({failure_rate:.2f}), so future similar cases should be reviewed more carefully."
        )

    elif review_rate >= 0.50 and total_action_score >= 0.70 and dynamic_severity >= 0.70:
        recommendation = "increase_escalation_clarity"
        action_bias = "keep_review_but_escalate_priority"
        intensity_adjustment = "increase_priority"
        confidence_adjustment = -0.03
        reason_codes.append(
            f"Recent review rate is high ({review_rate:.2f}) while current risk is also high, so review priority should increase."
        )

    elif success_rate >= 0.75 and matched_count >= 5:
        recommendation = "trust_current_route"
        action_bias = "allow_same_route_if_current_signals_match"
        intensity_adjustment = "maintain_or_slightly_increase"
        confidence_adjustment = 0.04
        reason_codes.append(
            f"Recent success rate is strong ({success_rate:.2f}), so the current route can be trusted for similar cases."
        )

    else:
        recommendation = "maintain_policy"
        action_bias = "keep_current_policy"
        intensity_adjustment = "none"
        confidence_adjustment = 0.0
        reason_codes.append(
            "Recent outcomes do not justify changing the current action policy."
        )

    reason_codes.append(f"Current action intensity was '{current_intensity}'.")

    return {
        "adaptation_available": True,
        "recommendation": recommendation,
        "action_bias": action_bias,
        "intensity_adjustment": intensity_adjustment,
        "confidence_adjustment": round(confidence_adjustment, 4),
        "reason_codes": reason_codes,
    }


# =========================================================
# Main Function
# =========================================================

def log_and_adapt_outcome(
    cypher_text: str,
    decoded_plain_text: str,
    predicted_class: str,
    svm_confidence: float,
    total_action_score: float,
    dynamic_severity: float,
    shap_cosine_similarity: float,
    rag_decision_result: Dict[str, Any],
    react_routing_result: Dict[str, Any],
    api_result: Dict[str, Any],
    action_required: bool,
    feedback: str = "",
    log_path: str | Path = DEFAULT_OUTCOME_LOG_PATH,
) -> Dict[str, Any]:
    """
    Logs the outcome of a live case and returns adaptation guidance.
    """

    timestamp = now_utc_iso()
    react_route = str(react_routing_result.get("route_id", ""))
    recommended_route = str(react_routing_result.get("recommended_route", ""))
    adaptive_action = react_routing_result.get("adaptive_action", {}) or {}

    outcome_status = infer_outcome_status(
        action_required=action_required,
        api_result=api_result,
        react_route=react_route,
        recommended_route=recommended_route,
    )

    case_id = stable_case_id(
        cypher_text=cypher_text,
        predicted_class=predicted_class,
        timestamp=timestamp,
    )

    history_before = load_outcome_history(log_path=log_path)

    recent_summary = summarize_recent_outcomes(
        history_df=history_before,
        predicted_class=predicted_class,
        react_route=react_route,
    )

    adaptation = compute_between_cycle_adaptation(
        recent_summary=recent_summary,
        total_action_score=total_action_score,
        dynamic_severity=dynamic_severity,
        current_intensity=str(adaptive_action.get("intensity", "unknown")),
    )

    row = {
        "timestamp_utc": timestamp,
        "case_id": case_id,
        "predicted_class": predicted_class,
        "svm_confidence": round(clamp01(svm_confidence), 4),
        "total_action_score": round(clamp01(total_action_score), 4),
        "dynamic_severity": round(clamp01(dynamic_severity), 4),
        "shap_cosine_similarity": round(clamp01(shap_cosine_similarity), 4),
        "rag_case_type": rag_decision_result.get("case_type"),
        "rag_used": bool(rag_decision_result.get("rag_used")),
        "rag_majority_vote": rag_decision_result.get("majority_vote"),
        "rag_recommended_route": rag_decision_result.get("recommended_route"),
        "react_route": react_route,
        "react_route_name": react_routing_result.get("route_name"),
        "react_recommended_route": recommended_route,
        "adaptive_action_type": adaptive_action.get("action_type"),
        "adaptive_timing": adaptive_action.get("timing"),
        "adaptive_intensity": adaptive_action.get("intensity"),
        "action_required": bool(action_required),
        "api_success": bool(api_result.get("success")),
        "api_message": api_result.get("message"),
        "outcome_status": outcome_status,
        "feedback": feedback,
        "decoded_plain_text_preview": str(decoded_plain_text or "")[:250],
        "cypher_text_preview": str(cypher_text or "")[:250],
        "adaptation_recommendation": adaptation.get("recommendation"),
        "adaptation_action_bias": adaptation.get("action_bias"),
        "adaptation_confidence_adjustment": adaptation.get("confidence_adjustment"),
        "adaptation_payload_json": json_dumps_safe(adaptation),
    }

    try:
        append_outcome_row(row=row, log_path=log_path)
        logged = True
        log_error = None
    except Exception as e:
        logged = False
        log_error = str(e)

    history_after = load_outcome_history(log_path=log_path)

    return {
        "success": bool(logged),
        "method": "outcome_tracking_and_between_cycle_adaptation",
        "case_id": case_id,
        "timestamp_utc": timestamp,
        "outcome_status": outcome_status,
        "logged": bool(logged),
        "log_path": str(log_path),
        "log_error": log_error,
        "recent_summary_before_current_case": recent_summary,
        "adaptation": adaptation,
        "history_count_after_logging": int(len(history_after)) if history_after is not None else 0,
        "observe_after_action": {
            "api_success": bool(api_result.get("success")),
            "api_message": api_result.get("message"),
            "react_route": react_route,
            "recommended_route": recommended_route,
            "outcome_status": outcome_status,
            "feedback": feedback or "No manual feedback provided yet.",
        },
    }
