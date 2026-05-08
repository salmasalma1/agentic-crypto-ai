# pipeline/outer_react_loop.py

"""
outer_react_loop.py

Formal Outer ReAct Rebuild Loop for CRYPTO AI.

Purpose:
- Trigger the rebuild loop only when strict dual drift is confirmed.
- Execute the required outer-cycle stages:
    1. Re-clustering
    2. Classifier retraining
    3. SHAP reference recomputation marker
    4. RAG index rebuild marker
    5. ARIMA/SARIMA baseline reset marker
    6. Rollback gating based on classifier quality
- Return a complete audit trail that can be shown in Streamlit and LangSmith.

Important:
This module is deliberately conservative. It creates a candidate rebuilt state and
uses rollback gating before accepting it. If the new classifier quality is below
the gate, the current live artifacts remain trusted.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pipeline.dynamic_reclustering import run_dynamic_reclustering


# =========================================================
# Helpers
# =========================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)

    if np.isnan(number) or np.isinf(number):
        return float(default)

    return number


def clamp01(value: Any, default: float = 0.0) -> float:
    number = safe_float(value, default=default)

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def extract_metric(metrics: Any, names: List[str], default: float = 0.0) -> float:
    """
    Extracts a metric from a dict-like object using several possible names.
    """

    if isinstance(metrics, dict):
        for name in names:
            if name in metrics:
                return safe_float(metrics.get(name), default=default)

    return float(default)


def dataframe_profile(df: pd.DataFrame, feature_cols: List[str], risk_feature: str) -> Dict[str, Any]:
    """
    Builds a lightweight baseline profile for ARIMA/cluster monitoring.
    """

    if df is None or df.empty:
        return {
            "rows": 0,
            "feature_count": len(feature_cols),
            "risk_feature": risk_feature,
            "risk_feature_mean": 0.0,
            "risk_feature_std": 0.0,
        }

    profile = {
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "risk_feature": risk_feature,
    }

    if risk_feature in df.columns:
        values = pd.to_numeric(df[risk_feature], errors="coerce").fillna(0.0)
        profile["risk_feature_mean"] = round(float(values.mean()), 6)
        profile["risk_feature_std"] = round(float(values.std()), 6)
        profile["risk_feature_min"] = round(float(values.min()), 6)
        profile["risk_feature_max"] = round(float(values.max()), 6)
    else:
        profile["risk_feature_mean"] = 0.0
        profile["risk_feature_std"] = 0.0
        profile["risk_feature_min"] = 0.0
        profile["risk_feature_max"] = 0.0

    return profile


def build_cluster_health_summary(reclustered_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarizes the rebuilt cluster distribution.
    """

    summary: Dict[str, Any] = {}

    if reclustered_df is None or reclustered_df.empty:
        return {
            "rows": 0,
            "dbscan_clusters": {},
            "fcm_clusters": {},
            "risk_labels": {},
        }

    summary["rows"] = int(len(reclustered_df))

    for column, key in [
        ("dbscan_cluster", "dbscan_clusters"),
        ("fcm_cluster", "fcm_clusters"),
        ("risk_label", "risk_labels"),
        ("business_risk_label", "business_risk_labels"),
    ]:
        if column in reclustered_df.columns:
            counts = reclustered_df[column].astype(str).value_counts().head(20)
            summary[key] = {str(k): int(v) for k, v in counts.items()}
        else:
            summary[key] = {}

    return summary


def build_shap_reference_manifest(reclustered_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Produces a concrete SHAP-reference recomputation manifest.

    In this live loop, SHAP local explanations are generated per live case, while
    this manifest rebuilds the cluster-level reference profile used by SHAP cosine
    reliability. It does not need to re-run expensive SHAP on every historical row.
    """

    if reclustered_df is None or reclustered_df.empty:
        return {
            "status": "not_recomputed",
            "reason": "No reclustered dataframe is available.",
            "reference_rows": 0,
        }

    cluster_column = "fcm_cluster" if "fcm_cluster" in reclustered_df.columns else "risk_label"

    available_features = [col for col in feature_cols if col in reclustered_df.columns]

    references = []

    if cluster_column in reclustered_df.columns and available_features:
        numeric_df = reclustered_df.copy()

        for col in available_features:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").fillna(0.0)

        grouped = numeric_df.groupby(cluster_column)

        for cluster_value, cluster_df in grouped:
            mean_profile = cluster_df[available_features].mean().abs()
            top_features = mean_profile.sort_values(ascending=False).head(10)

            references.append(
                {
                    "cluster": str(cluster_value),
                    "rows": int(len(cluster_df)),
                    "top_reference_features": [
                        {
                            "feature": str(feature),
                            "reference_weight": round(float(value), 6),
                        }
                        for feature, value in top_features.items()
                    ],
                }
            )

    return {
        "status": "recomputed",
        "method": "cluster_level_feature_profile_for_shap_cosine_reference",
        "cluster_column": cluster_column,
        "reference_rows": int(len(reclustered_df)),
        "cluster_reference_count": len(references),
        "references_preview": references[:10],
    }


def build_rag_rebuild_manifest(reclustered_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Creates a concrete RAG rebuild manifest for the KNN/RAG layer.

    The existing project uses a KNN-style retrieval layer. This manifest records
    the candidate rows and vector columns that should back the refreshed retrieval
    index after rollback gating is accepted.
    """

    rows = int(len(reclustered_df)) if reclustered_df is not None else 0
    available_features = []

    if reclustered_df is not None and not reclustered_df.empty:
        available_features = [col for col in feature_cols if col in reclustered_df.columns]

    return {
        "status": "rebuilt_manifest_ready" if rows > 0 and available_features else "not_rebuilt",
        "method": "knn_feature_and_shap_reference_manifest",
        "indexed_rows": rows,
        "vector_feature_count": len(available_features),
        "vector_features_preview": available_features[:20],
        "note": (
            "The retrieval manifest is refreshed from the reclustered candidate dataset. "
            "It is accepted only if classifier rollback gating passes."
        ),
    }


def write_outer_loop_audit(audit_payload: Dict[str, Any], artifacts_dir: str = "artifacts") -> str:
    """
    Writes an audit JSON file for the outer loop. Failure is non-fatal.
    """

    try:
        path = Path(artifacts_dir)
        path.mkdir(parents=True, exist_ok=True)

        audit_path = path / "outer_react_loop_audit.json"

        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2, ensure_ascii=False, default=str)

        return str(audit_path)

    except Exception:
        return ""


# =========================================================
# Main Outer ReAct Loop
# =========================================================

def run_outer_react_rebuild_loop(
    training_labeled_df: pd.DataFrame,
    training_scaled: np.ndarray,
    feature_cols: List[str],
    risk_feature: str,
    dual_drift_result: Dict[str, Any],
    current_metadata: Dict[str, Any] | None = None,
    current_svm_accuracy: float | None = None,
    enable_rebuild: bool = True,
    min_classifier_accuracy: float = 0.70,
    max_allowed_accuracy_drop: float = 0.10,
) -> Dict[str, Any]:
    """
    Executes the formal outer ReAct rebuild loop.

    The loop only triggers when dual drift requires an outer loop.
    If a candidate rebuilt classifier fails quality gates, rollback is enforced.
    """

    current_metadata = current_metadata or {}

    outer_loop_required = bool(dual_drift_result.get("outer_loop_required"))
    dual_drift_detected = bool(dual_drift_result.get("dual_drift_detected"))

    observe_state = {
        "dual_drift_detected": dual_drift_detected,
        "outer_loop_required": outer_loop_required,
        "dual_drift_decision": dual_drift_result.get("decision"),
        "signals": dual_drift_result.get("signals", {}),
    }

    if not outer_loop_required:
        return {
            "success": True,
            "method": "outer_react_rebuild_loop",
            "outer_loop_triggered": False,
            "decision": "OUTER_LOOP_NOT_TRIGGERED",
            "status": (
                "Outer ReAct loop was not triggered because strict dual drift was not confirmed."
            ),
            "rollback_required": False,
            "accepted_candidate": False,
            "react_trace": {
                "observe": observe_state,
                "reason": (
                    "At least one of the required independent drift signals is missing; "
                    "single-signal drift is treated as noise."
                ),
                "act": "No rebuild action executed.",
                "observe_after_action": {
                    "system_state": "current_artifacts_preserved",
                },
            },
        }

    if not enable_rebuild:
        return {
            "success": True,
            "method": "outer_react_rebuild_loop",
            "outer_loop_triggered": False,
            "decision": "OUTER_LOOP_DISABLED",
            "status": "Dual drift was detected, but rebuild is disabled by runtime control.",
            "rollback_required": False,
            "accepted_candidate": False,
            "react_trace": {
                "observe": observe_state,
                "reason": "Dual drift requires rebuild, but runtime setting disabled the rebuild action.",
                "act": "No rebuild action executed.",
                "observe_after_action": {
                    "system_state": "current_artifacts_preserved",
                },
            },
        }

    current_accuracy = current_svm_accuracy

    if current_accuracy is None:
        current_accuracy = extract_metric(
            current_metadata,
            names=["svm_accuracy", "accuracy", "model_accuracy"],
            default=0.0,
        )

    current_accuracy = clamp01(current_accuracy, default=0.0)

    started_at = datetime.utcnow().isoformat()

    try:
        # -------------------------------------------------
        # Act 1: Re-cluster + retrain candidate classifier.
        # -------------------------------------------------
        rebuild_result = run_dynamic_reclustering(
            training_labeled_df,
            training_scaled,
            feature_cols,
            risk_feature,
        )

        reclustered_df = rebuild_result.get("reclustered_df")
        new_svm_metrics = rebuild_result.get("svm_metrics", {})

        new_accuracy = extract_metric(
            new_svm_metrics,
            names=["accuracy", "test_accuracy", "train_accuracy", "macro_accuracy"],
            default=0.0,
        )

        new_accuracy = clamp01(new_accuracy, default=0.0)

        if current_accuracy > 0:
            accuracy_drop = max(0.0, current_accuracy - new_accuracy)
        else:
            accuracy_drop = 0.0

        # -------------------------------------------------
        # Act 2: Recompute dependent references/manifests.
        # -------------------------------------------------
        cluster_health = build_cluster_health_summary(reclustered_df)
        shap_reference_manifest = build_shap_reference_manifest(reclustered_df, feature_cols)
        rag_rebuild_manifest = build_rag_rebuild_manifest(reclustered_df, feature_cols)
        arima_baseline_manifest = {
            "status": "reset_ready",
            "method": "risk_feature_baseline_profile",
            "baseline": dataframe_profile(reclustered_df, feature_cols, risk_feature),
        }

        # -------------------------------------------------
        # Act 3: Rollback gating.
        # -------------------------------------------------
        quality_gate_passed = True
        gate_reasons = []

        if new_accuracy < min_classifier_accuracy:
            quality_gate_passed = False
            gate_reasons.append(
                f"New classifier accuracy {new_accuracy:.4f} is below minimum gate {min_classifier_accuracy:.4f}."
            )

        if current_accuracy > 0 and accuracy_drop > max_allowed_accuracy_drop:
            quality_gate_passed = False
            gate_reasons.append(
                f"Accuracy drop {accuracy_drop:.4f} exceeds allowed drop {max_allowed_accuracy_drop:.4f}."
            )

        if not gate_reasons:
            gate_reasons.append("Candidate rebuild passed classifier quality gates.")

        accepted_candidate = bool(quality_gate_passed)
        rollback_required = not accepted_candidate

        decision = "ACCEPT_REBUILT_CANDIDATE" if accepted_candidate else "ROLLBACK_TO_PREVIOUS_ARTIFACTS"

        finished_at = datetime.utcnow().isoformat()

        audit_payload = {
            "method": "outer_react_rebuild_loop",
            "started_at": started_at,
            "finished_at": finished_at,
            "decision": decision,
            "outer_loop_triggered": True,
            "dual_drift_result": dual_drift_result,
            "current_accuracy": round(current_accuracy, 4),
            "new_accuracy": round(new_accuracy, 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "quality_gate_passed": quality_gate_passed,
            "gate_reasons": gate_reasons,
            "cluster_health": cluster_health,
            "shap_reference_manifest": shap_reference_manifest,
            "rag_rebuild_manifest": rag_rebuild_manifest,
            "arima_baseline_manifest": arima_baseline_manifest,
        }

        audit_path = write_outer_loop_audit(audit_payload)

        return {
            "success": True,
            "method": "outer_react_rebuild_loop",
            "outer_loop_triggered": True,
            "decision": decision,
            "status": (
                "Outer ReAct rebuild completed and candidate was accepted."
                if accepted_candidate
                else "Outer ReAct rebuild completed but rollback gating rejected the candidate."
            ),
            "accepted_candidate": accepted_candidate,
            "rollback_required": rollback_required,
            "rollback_reason": "; ".join(gate_reasons),
            "current_accuracy": round(current_accuracy, 4),
            "new_accuracy": round(new_accuracy, 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "quality_gate_passed": quality_gate_passed,
            "gate_reasons": gate_reasons,
            "rebuild_actions": {
                "reclustering": "completed",
                "classifier_retraining": "completed",
                "shap_recomputation": shap_reference_manifest.get("status"),
                "rag_index_rebuild": rag_rebuild_manifest.get("status"),
                "arima_baseline_reset": arima_baseline_manifest.get("status"),
            },
            "cluster_health": cluster_health,
            "shap_reference_manifest": shap_reference_manifest,
            "rag_rebuild_manifest": rag_rebuild_manifest,
            "arima_baseline_manifest": arima_baseline_manifest,
            "candidate_rebuild_result": rebuild_result if accepted_candidate else {},
            "audit_path": audit_path,
            "react_trace": {
                "observe": observe_state,
                "reason": (
                    "Both independent drift signals fired. The system must rebuild a "
                    "candidate model stack and apply rollback gating before accepting it."
                ),
                "act": {
                    "reclustering": "DBSCAN and FCM rerun",
                    "classifier_retraining": "SVM retrained on relabeled candidate data",
                    "shap_recomputation": shap_reference_manifest.get("status"),
                    "rag_index_rebuild": rag_rebuild_manifest.get("status"),
                    "arima_baseline_reset": arima_baseline_manifest.get("status"),
                    "rollback_gating": decision,
                },
                "observe_after_action": {
                    "accepted_candidate": accepted_candidate,
                    "rollback_required": rollback_required,
                    "current_accuracy": round(current_accuracy, 4),
                    "new_accuracy": round(new_accuracy, 4),
                    "gate_reasons": gate_reasons,
                    "audit_path": audit_path,
                },
            },
        }

    except Exception as e:
        return {
            "success": False,
            "method": "outer_react_rebuild_loop",
            "outer_loop_triggered": True,
            "decision": "OUTER_LOOP_ERROR_ROLLBACK",
            "status": "Outer ReAct rebuild failed; rollback to previous artifacts is required.",
            "accepted_candidate": False,
            "rollback_required": True,
            "rollback_reason": str(e),
            "error": str(e),
            "react_trace": {
                "observe": observe_state,
                "reason": "Dual drift required rebuild, but the rebuild process raised an error.",
                "act": "Attempted rebuild; error occurred.",
                "observe_after_action": {
                    "rollback_required": True,
                    "error": str(e),
                    "system_state": "previous_artifacts_preserved",
                },
            },
        }
