# pipeline/rag_decision_router.py

"""
rag_decision_router.py

Explicit RAG Majority Vote module for CRYPTO AI.

Purpose:
- Use RAG only when the case is borderline.
- Read retrieved RAG items.
- Extract decision votes from retrieved evidence.
- Perform majority vote.
- Detect RAG majority, RAG tie, and RAG zero-success / no useful evidence.
- Compare RAG vote with ARIMA/SARIMA trend.
- Return a routing-ready result for the ReAct decision engine.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple


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


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def is_borderline_score(total_action_score: float, lower: float = 0.55, upper: float = 0.75) -> bool:
    score = clamp01(total_action_score)
    return lower <= score < upper


def extract_text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)

    possible_keys = [
        "text", "content", "context", "document", "page_content", "snippet",
        "retrieved_context", "description", "reason", "message", "label",
        "action", "outcome", "recommendation",
    ]
    collected = []
    for key in possible_keys:
        value = item.get(key)
        if value:
            collected.append(str(value))

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if value:
                collected.append(f"{key}: {value}")

    return " ".join(collected)


def infer_vote_from_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "unknown"

    dispatch_terms = [
        "dispatch", "send alert", "trigger alert", "api action", "email alert",
        "action required", "high risk", "critical", "escalate", "block", "notify", "soc",
    ]
    review_terms = [
        "human review", "manual review", "analyst review", "low confidence",
        "uncertain", "investigate", "needs review", "ambiguous",
    ]
    monitor_terms = [
        "monitor", "no action", "safe", "low risk", "below threshold", "observe only", "watch",
    ]
    shap_terms = [
        "shap recheck", "recheck shap", "explanation unreliable", "low cosine",
        "low shap reliability", "fallback to shap",
    ]

    if any(term in text for term in dispatch_terms):
        return "dispatch"
    if any(term in text for term in review_terms):
        return "human_review"
    if any(term in text for term in shap_terms):
        return "shap_recheck"
    if any(term in text for term in monitor_terms):
        return "monitor"
    return "unknown"


def extract_vote_from_item(item: Any) -> Tuple[str, str]:
    if isinstance(item, dict):
        explicit_fields = ["vote", "decision", "action", "recommendation", "route", "outcome", "label"]
        for field in explicit_fields:
            value = item.get(field)
            if value:
                normalized = normalize_text(value)
                if normalized in {"dispatch", "human_review", "monitor", "shap_recheck"}:
                    return normalized, extract_text_from_item(item)
                if normalized in {"action_required", "alert", "escalate"}:
                    return "dispatch", extract_text_from_item(item)
                if normalized in {"review", "manual_review", "analyst_review"}:
                    return "human_review", extract_text_from_item(item)
                if normalized in {"no_action", "observe", "watch"}:
                    return "monitor", extract_text_from_item(item)

    evidence_text = extract_text_from_item(item)
    vote = infer_vote_from_text(evidence_text)
    return vote, evidence_text


def summarize_votes(votes: List[str]) -> Dict[str, Any]:
    valid_votes = [v for v in votes if v in {"dispatch", "human_review", "monitor", "shap_recheck"}]
    if not valid_votes:
        return {
            "valid_vote_count": 0,
            "vote_counts": {},
            "majority_vote": "none",
            "majority_confidence": 0.0,
            "tie_detected": False,
            "zero_success": True,
        }

    counts = Counter(valid_votes)
    most_common = counts.most_common()
    top_vote, top_count = most_common[0]
    tie_detected = len(most_common) > 1 and top_count == most_common[1][1]
    majority_confidence = top_count / len(valid_votes)

    return {
        "valid_vote_count": len(valid_votes),
        "vote_counts": dict(counts),
        "majority_vote": "tie" if tie_detected else top_vote,
        "majority_confidence": round(float(majority_confidence), 4),
        "tie_detected": bool(tie_detected),
        "zero_success": False,
    }


def analyze_arima_rag_alignment(majority_vote: str, forecast_risk_score: float, expected_change_pct: float) -> Dict[str, Any]:
    forecast_risk_score = clamp01(forecast_risk_score)
    try:
        expected_change_pct = float(expected_change_pct)
    except Exception:
        expected_change_pct = 0.0

    rising_trend = expected_change_pct >= 5 or forecast_risk_score >= 0.65
    declining_trend = expected_change_pct <= -5 or forecast_risk_score <= 0.30

    if majority_vote == "dispatch":
        if rising_trend:
            return {
                "arima_alignment": "supports_rag",
                "arima_vote": "trust_rag",
                "trend_direction": "rising",
                "confidence_multiplier": 1.15,
                "reason": "ARIMA/SARIMA trend is rising or forecast risk is high, so it supports RAG dispatch.",
            }
        if declining_trend:
            return {
                "arima_alignment": "against_rag",
                "arima_vote": "dampen_to_review",
                "trend_direction": "declining",
                "confidence_multiplier": 0.75,
                "reason": "ARIMA/SARIMA trend is declining or forecast risk is low, so dispatch is dampened toward review.",
            }

    if majority_vote in {"monitor", "human_review", "shap_recheck"}:
        if rising_trend:
            return {
                "arima_alignment": "against_rag",
                "arima_vote": "raise_caution",
                "trend_direction": "rising",
                "confidence_multiplier": 0.90,
                "reason": "RAG vote is conservative, but ARIMA/SARIMA indicates rising risk.",
            }
        if declining_trend:
            return {
                "arima_alignment": "supports_rag",
                "arima_vote": "trust_rag",
                "trend_direction": "declining",
                "confidence_multiplier": 1.05,
                "reason": "RAG vote is conservative and ARIMA/SARIMA trend is declining, so RAG is supported.",
            }

    return {
        "arima_alignment": "neutral",
        "arima_vote": "neutral",
        "trend_direction": "stable",
        "confidence_multiplier": 1.00,
        "reason": "ARIMA/SARIMA trend does not strongly support or oppose the RAG vote.",
    }


def final_rag_recommendation(
    majority_vote: str,
    majority_confidence: float,
    tie_detected: bool,
    zero_success: bool,
    arima_alignment: Dict[str, Any],
    shap_cosine_similarity: float,
) -> Dict[str, Any]:
    shap_cosine_similarity = clamp01(shap_cosine_similarity)

    if zero_success:
        return {
            "rag_route": "rag_zero_success",
            "recommended_route": "shap_recheck" if shap_cosine_similarity < 0.65 else "human_review",
            "action_type": "review_or_recheck",
            "reason": "RAG retrieved no useful voting evidence, so the system falls back to SHAP reliability or human review.",
        }

    if tie_detected:
        return {
            "rag_route": "rag_tie",
            "recommended_route": "human_review",
            "action_type": "manual_review",
            "reason": "RAG evidence produced a tie, so the safest route is human review.",
        }

    if majority_vote == "dispatch":
        if arima_alignment.get("arima_vote") == "dampen_to_review":
            return {
                "rag_route": "rag_majority_dispatch_arima_dampened",
                "recommended_route": "human_review",
                "action_type": "manual_review",
                "reason": "RAG voted dispatch, but declining ARIMA trend dampened the decision toward human review.",
            }
        if majority_confidence >= 0.60:
            return {
                "rag_route": "rag_majority_dispatch",
                "recommended_route": "dispatch",
                "action_type": "api_alert",
                "reason": "RAG majority supports dispatch with sufficient confidence.",
            }
        return {
            "rag_route": "rag_weak_dispatch",
            "recommended_route": "human_review",
            "action_type": "manual_review",
            "reason": "RAG supports dispatch weakly, so human review is safer.",
        }

    if majority_vote == "human_review":
        return {
            "rag_route": "rag_majority_review",
            "recommended_route": "human_review",
            "action_type": "manual_review",
            "reason": "RAG majority recommends human review.",
        }

    if majority_vote == "shap_recheck":
        return {
            "rag_route": "rag_majority_shap_recheck",
            "recommended_route": "shap_recheck",
            "action_type": "model_recheck",
            "reason": "RAG majority recommends SHAP reliability re-check.",
        }

    if majority_vote == "monitor":
        if arima_alignment.get("arima_vote") == "raise_caution":
            return {
                "rag_route": "rag_monitor_arima_caution",
                "recommended_route": "human_review",
                "action_type": "manual_review",
                "reason": "RAG suggests monitoring, but rising ARIMA risk raises the case to review.",
            }
        return {
            "rag_route": "rag_majority_monitor",
            "recommended_route": "monitor",
            "action_type": "monitor_only",
            "reason": "RAG majority recommends monitoring without immediate action.",
        }

    return {
        "rag_route": "rag_unknown",
        "recommended_route": "human_review",
        "action_type": "manual_review",
        "reason": "RAG result is unknown, so the system routes to human review.",
    }


def run_explicit_rag_majority_vote(
    rag_result: Dict[str, Any],
    total_action_score: float,
    svm_confidence: float,
    shap_cosine_similarity: float,
    forecast_risk_score: float,
    expected_change_pct: float,
    lower_borderline: float = 0.55,
    upper_borderline: float = 0.75,
) -> Dict[str, Any]:
    total_action_score = clamp01(total_action_score)
    svm_confidence = clamp01(svm_confidence)
    shap_cosine_similarity = clamp01(shap_cosine_similarity)
    forecast_risk_score = clamp01(forecast_risk_score)

    borderline = is_borderline_score(total_action_score, lower_borderline, upper_borderline)

    if not borderline:
        return {
            "success": True,
            "rag_used": False,
            "case_type": "not_borderline",
            "reason": "Explicit RAG majority vote was not used because the total action score is outside the configured borderline range.",
            "borderline_range": {"lower": lower_borderline, "upper": upper_borderline},
            "total_action_score": round(total_action_score, 4),
            "recommended_route": "not_applicable",
            "majority_vote": "not_applicable",
            "vote_counts": {},
            "tie_detected": False,
            "zero_success": False,
        }

    retrieved_items = []
    retrieved_context = ""
    if isinstance(rag_result, dict):
        retrieved_items = rag_result.get("retrieved_items") or rag_result.get("items") or rag_result.get("documents") or []
        retrieved_context = rag_result.get("retrieved_context", "")

    votes = []
    evidence_rows = []

    for idx, item in enumerate(retrieved_items, start=1):
        vote, evidence_text = extract_vote_from_item(item)
        votes.append(vote)
        evidence_rows.append({"rank": idx, "vote": vote, "evidence_preview": evidence_text[:350]})

    if not votes and retrieved_context:
        vote = infer_vote_from_text(retrieved_context)
        votes.append(vote)
        evidence_rows.append({"rank": 1, "vote": vote, "evidence_preview": retrieved_context[:350]})

    vote_summary = summarize_votes(votes)
    arima_alignment = analyze_arima_rag_alignment(
        majority_vote=vote_summary["majority_vote"],
        forecast_risk_score=forecast_risk_score,
        expected_change_pct=expected_change_pct,
    )
    recommendation = final_rag_recommendation(
        majority_vote=vote_summary["majority_vote"],
        majority_confidence=vote_summary["majority_confidence"],
        tie_detected=vote_summary["tie_detected"],
        zero_success=vote_summary["zero_success"],
        arima_alignment=arima_alignment,
        shap_cosine_similarity=shap_cosine_similarity,
    )
    adjusted_vote_confidence = clamp01(
        vote_summary["majority_confidence"] * float(arima_alignment.get("confidence_multiplier", 1.0))
    )

    return {
        "success": True,
        "rag_used": True,
        "case_type": "borderline",
        "method": "explicit_rag_majority_vote",
        "borderline_range": {"lower": lower_borderline, "upper": upper_borderline},
        "inputs": {
            "total_action_score": round(total_action_score, 4),
            "svm_confidence": round(svm_confidence, 4),
            "shap_cosine_similarity": round(shap_cosine_similarity, 4),
            "forecast_risk_score": round(forecast_risk_score, 4),
            "expected_change_pct": expected_change_pct,
        },
        "retrieved_count": len(retrieved_items),
        "valid_vote_count": vote_summary["valid_vote_count"],
        "votes": votes,
        "vote_counts": vote_summary["vote_counts"],
        "majority_vote": vote_summary["majority_vote"],
        "majority_confidence": vote_summary["majority_confidence"],
        "adjusted_vote_confidence": round(adjusted_vote_confidence, 4),
        "tie_detected": vote_summary["tie_detected"],
        "zero_success": vote_summary["zero_success"],
        "arima_alignment": arima_alignment,
        "recommended_route": recommendation["recommended_route"],
        "rag_route": recommendation["rag_route"],
        "action_type": recommendation["action_type"],
        "reason": recommendation["reason"],
        "evidence_rows": evidence_rows,
    }
