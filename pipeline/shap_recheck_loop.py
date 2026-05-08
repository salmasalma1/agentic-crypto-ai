# pipeline/shap_recheck_loop.py

"""
shap_recheck_loop.py

Formal SHAP Re-check Loop for CRYPTO AI.

Purpose:
- If SHAP cosine reliability is below the reliability threshold,
  run a second SHAP explanation pass using a stricter/larger background.
- Recalculate SHAP cosine reliability after the re-check.
- Decide whether the case can continue or must be routed to human review.

This closes the ReAct path:

    R2_SHAP_RECHECK_LOW_RELIABILITY

The loop is not a placeholder. It performs:
1. Observe low SHAP cosine.
2. Reason that explanation reliability is weak.
3. Act by re-running SHAP explanation.
4. Observe the updated SHAP cosine reliability.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pipeline.shap_explainer import run_shap_explanation
from pipeline.shap_reliability import calculate_shap_cluster_reliability


# =========================================================
# Helpers
# =========================================================

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


def reliability_level(score: float) -> str:
    score = clamp01(score)

    if score >= 0.85:
        return "high"

    if score >= 0.65:
        return "medium"

    return "low"


def safe_importance_df(shap_result: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    """
    Ensures a valid SHAP importance dataframe exists.
    """

    importance_df = shap_result.get("importance_df")

    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        return importance_df

    return pd.DataFrame(
        {
            "feature": feature_cols,
            "shap_importance": [0.0] * len(feature_cols),
        }
    )


# =========================================================
# Main SHAP Re-check Loop
# =========================================================

def run_shap_recheck_loop(
    svm_model,
    live_scaled: np.ndarray,
    shap_background_scaled: np.ndarray,
    feature_cols: List[str],
    training_labeled_df: pd.DataFrame,
    live_df: pd.DataFrame,
    predicted_class: str,
    original_shap_result: Dict[str, Any],
    original_shap_cosine_similarity: float,
    reliability_threshold: float = 0.65,
    max_background_samples: int = 120,
) -> Dict[str, Any]:
    """
    Runs the formal SHAP re-check loop.

    Parameters:
    - svm_model: saved SVM model
    - live_scaled: scaled live row, shape (1, n_features)
    - shap_background_scaled: scaled SHAP background rows
    - feature_cols: saved feature columns
    - training_labeled_df: saved training dataframe
    - live_df: current live dataframe
    - predicted_class: SVM prediction
    - original_shap_result: first SHAP explanation result
    - original_shap_cosine_similarity: first cosine reliability score
    - reliability_threshold: minimum acceptable cosine reliability

    Returns a dict containing the final reliability, updated SHAP outputs, and
    the Observe -> Reason -> Act -> Observe trace for this loop.
    """

    original_cosine = clamp01(original_shap_cosine_similarity)
    threshold = clamp01(reliability_threshold, default=0.65)

    # -----------------------------------------------------
    # If SHAP is already reliable, do not rerun SHAP.
    # -----------------------------------------------------
    if original_cosine >= threshold:
        return {
            "success": True,
            "method": "shap_recheck_loop",
            "recheck_triggered": False,
            "reason": "SHAP cosine reliability is already above threshold; re-check was not required.",
            "original_shap_cosine_similarity": round(original_cosine, 4),
            "final_shap_cosine_similarity": round(original_cosine, 4),
            "final_reliability_level": reliability_level(original_cosine),
            "route_after_recheck": "continue_current_route",
            "updated_shap_result": original_shap_result,
            "updated_shap_reliability_result": original_shap_result.get(
                "shap_reliability_result",
                {},
            ),
            "react_loop": {
                "observe": {
                    "original_shap_cosine_similarity": round(original_cosine, 4),
                    "threshold": threshold,
                },
                "reason": "Reliability is acceptable.",
                "act": "No SHAP re-check executed.",
                "observe_after_action": {
                    "final_reliability": round(original_cosine, 4),
                    "decision": "continue_current_route",
                },
            },
        }

    # -----------------------------------------------------
    # SHAP is unreliable: run second SHAP pass.
    # -----------------------------------------------------
    try:
        live_scaled = np.asarray(live_scaled, dtype=float)
        shap_background_scaled = np.asarray(shap_background_scaled, dtype=float)

        if live_scaled.ndim == 1:
            live_scaled = live_scaled.reshape(1, -1)

        if shap_background_scaled.ndim == 1:
            shap_background_scaled = shap_background_scaled.reshape(1, -1)

        # Put the live row first so max_explain_samples=1 explains it.
        X_for_recheck = np.vstack(
            [
                live_scaled,
                shap_background_scaled,
            ]
        )

        # Re-run SHAP with a larger/more conservative background.
        updated_shap_result = run_shap_explanation(
            svm_model,
            X_for_recheck,
            feature_cols,
            max_background_samples=min(max_background_samples, len(X_for_recheck)),
            max_explain_samples=1,
        )

        updated_importance_df = safe_importance_df(
            updated_shap_result,
            feature_cols,
        )

        updated_reliability_result = calculate_shap_cluster_reliability(
            importance_df=updated_importance_df,
            feature_cols=feature_cols,
            training_labeled_df=training_labeled_df,
            live_df=live_df,
            predicted_class=predicted_class,
        )

        updated_cosine = clamp01(
            updated_reliability_result.get(
                "shap_cosine_similarity",
                0.0,
            )
        )

        updated_level = reliability_level(updated_cosine)

        updated_shap_result["shap_cosine_similarity"] = updated_cosine
        updated_shap_result["shap_reliability_result"] = updated_reliability_result

        if updated_cosine >= threshold:
            route_after_recheck = "continue_after_successful_shap_recheck"
            decision_reason = (
                "SHAP reliability improved above threshold after re-check, "
                "so the system can continue routing."
            )
        else:
            route_after_recheck = "human_review_after_failed_shap_recheck"
            decision_reason = (
                "SHAP reliability remained below threshold after re-check, "
                "so the system must route to human review."
            )

        return {
            "success": True,
            "method": "shap_recheck_loop",
            "recheck_triggered": True,
            "reason": decision_reason,
            "original_shap_cosine_similarity": round(original_cosine, 4),
            "final_shap_cosine_similarity": round(updated_cosine, 4),
            "final_reliability_level": updated_level,
            "route_after_recheck": route_after_recheck,
            "updated_shap_result": updated_shap_result,
            "updated_shap_reliability_result": updated_reliability_result,
            "react_loop": {
                "observe": {
                    "original_shap_cosine_similarity": round(original_cosine, 4),
                    "threshold": threshold,
                    "predicted_class": predicted_class,
                },
                "reason": (
                    "SHAP cosine reliability is below threshold, so the "
                    "explanation is not trusted enough for immediate action."
                ),
                "act": (
                    "Re-ran SHAP explanation using a larger/more conservative "
                    "background sample."
                ),
                "observe_after_action": {
                    "final_shap_cosine_similarity": round(updated_cosine, 4),
                    "final_reliability_level": updated_level,
                    "decision": route_after_recheck,
                },
            },
        }

    except Exception as e:
        return {
            "success": False,
            "method": "shap_recheck_loop",
            "recheck_triggered": True,
            "reason": "SHAP re-check failed; route to human review for safety.",
            "error": str(e),
            "original_shap_cosine_similarity": round(original_cosine, 4),
            "final_shap_cosine_similarity": round(original_cosine, 4),
            "final_reliability_level": reliability_level(original_cosine),
            "route_after_recheck": "human_review_after_recheck_error",
            "updated_shap_result": original_shap_result,
            "updated_shap_reliability_result": original_shap_result.get(
                "shap_reliability_result",
                {},
            ),
            "react_loop": {
                "observe": {
                    "original_shap_cosine_similarity": round(original_cosine, 4),
                    "threshold": threshold,
                    "predicted_class": predicted_class,
                },
                "reason": "SHAP reliability was low and required a re-check.",
                "act": "Attempted SHAP re-check.",
                "observe_after_action": {
                    "error": str(e),
                    "decision": "human_review_after_recheck_error",
                },
            },
        }
