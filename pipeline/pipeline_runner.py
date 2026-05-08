# pipeline/pipeline_runner.py

"""End-to-end live Sense-Reason-Act pipeline runner for CRYPTO AI.

This version separates:

1) Offline Training:
   - Run once using:
     python train_local_pipeline.py
   - Saves scaler, SVM-RBF, DBSCAN, FCM info, feature columns, SHAP background,
     and training dataframe into artifacts/.

2) Live Inference:
   - Streamlit calls run_crypto_ai_pipeline().
   - It loads saved artifacts.
   - It does NOT retrain SVM on the full dataset every run.
   - It only processes the new live message and predicts using the saved SVM.

Pipeline:
Sense:
- LLM/Regex cypher text extraction
- Data/artifact loading

Reason:
- Live feature engineering
- Saved scaler transform
- Saved SVM-RBF prediction
- Deterministic decoding after SVM
- RAG query preparation and mandatory explicit RAG decision check
- SHAP explanation
- SHAP cluster-mean cosine reliability
- SHAP English Explanation Node
- SARIMA forecasting
- Basic forecast drift detection
- Strict dual drift detection: Gap(SVM) decline + centroid movement
- Dynamic reclustering only if both drift signals fire

Act:
- Exact Five-Signal Intelligent Score with Dynamic Severity
- LLM/Fallback final report
- Email/API action
"""

from __future__ import annotations

import os
import base64
import binascii
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from actions.email_action import send_email_alert
from agent.cipher_text_extractor import extract_cypher_text
from agent.decoders_utils import decode_by_type
from agent.llm_reporter import generate_crypto_ai_report
from pipeline.action_score import calculate_total_action_score
from pipeline.cipher_feature_engineering import add_cipher_features
from pipeline.drift_detection import detect_drift
from pipeline.dynamic_reclustering import run_dynamic_reclustering
from pipeline.dynamic_severity import calculate_dynamic_severity
from pipeline.dual_drift_detector import detect_dual_drift
from pipeline.ingestion import load_data
from pipeline.model_persistence import (
    get_artifact_status,
    load_training_artifacts,
)
from pipeline.preprocessing import prepare_features
from pipeline.rag_knn_retriever import build_cipher_rag_query, retrieve_rag_context
from pipeline.rag_decision_router import run_explicit_rag_majority_vote
from pipeline.react_routing_engine import run_react_routing_engine
from pipeline.outcome_tracker import log_and_adapt_outcome
from pipeline.outer_react_loop import run_outer_react_rebuild_loop
from pipeline.sarima_forecasting import run_sarima_forecast
from pipeline.shap_english_explainer import generate_shap_english_explanation
from pipeline.shap_explainer import run_shap_explanation
from pipeline.shap_reliability import calculate_shap_cluster_reliability
from pipeline.shap_recheck_loop import run_shap_recheck_loop
from pipeline.svm_classifier import predict_with_confidence

try:
    from utils.config import SUPPORTED_CIPHER_TYPES
except Exception:
    SUPPORTED_CIPHER_TYPES = ["Base64", "Base32", "Morse", "Hex", "ASCII"]


# ==============================
# Manual LangSmith Tracing
# ==============================

try:
    import langsmith as ls  # type: ignore

    LANGSMITH_MANUAL_TRACE_AVAILABLE = True
except Exception:
    ls = None
    LANGSMITH_MANUAL_TRACE_AVAILABLE = False


class DummyTrace:
    """No-op trace object when LangSmith is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def end(self, outputs=None):
        return None


def _safe_for_langsmith(value: Any) -> Any:
    """
    Converts large objects into lightweight summaries for LangSmith.
    """

    try:
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(value.shape),
                "columns_sample": [str(col) for col in value.columns[:20]],
            }

        if isinstance(value, pd.Series):
            return {
                "type": "Series",
                "length": int(len(value)),
                "name": str(value.name),
            }

        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
            }

        if hasattr(value, "shape"):
            shape = getattr(value, "shape", None)
            return {
                "type": value.__class__.__name__,
                "shape": list(shape) if shape is not None else None,
            }

        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key in {
                    "raw_df",
                    "working_df",
                    "expanded_df",
                    "cleaned_df",
                    "labeled_df",
                    "training_labeled_df",
                    "live_df",
                }:
                    output[key] = _safe_for_langsmith(item)
                elif key in {"svm_model", "dbscan_model", "scaler"}:
                    output[key] = item.__class__.__name__ if item is not None else None
                elif key in {"forecast_df", "importance_df"}:
                    output[key] = _safe_for_langsmith(item)
                else:
                    output[key] = _safe_for_langsmith(item)
            return output

        if isinstance(value, (list, tuple)):
            if len(value) > 25:
                return {
                    "type": type(value).__name__,
                    "length": len(value),
                    "sample": [_safe_for_langsmith(item) for item in value[:10]],
                }
            return [_safe_for_langsmith(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    except Exception:
        return str(value)


def ls_trace(name: str, run_type: str = "chain", inputs: Dict[str, Any] | None = None):
    """
    Manual LangSmith trace wrapper.

    This is needed because sklearn, pandas, statsmodels, SHAP, and local Python
    code are not automatically visible in LangSmith unless manually traced.
    """

    if LANGSMITH_MANUAL_TRACE_AVAILABLE and os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")

        return ls.trace(
            name=name,
            run_type=run_type,
            project_name=os.getenv(
                "LANGSMITH_PROJECT",
                "cryptai-autonomous-pipeline",
            ),
            inputs=_safe_for_langsmith(inputs or {}),
        )

    return DummyTrace()


def end_trace(run, outputs: Dict[str, Any] | None = None) -> None:
    """
    Safely ends a LangSmith trace span.
    """

    try:
        run.end(outputs=_safe_for_langsmith(outputs or {}))
    except Exception:
        pass


# ==============================
# Helpers
# ==============================

def risk_weight_from_label(label: str) -> float:
    """
    Business-risk prior used in the Total Action Score.
    """

    label = str(label or "").lower()

    if "anomaly" in label or "morse" in label:
        return 0.80

    if "base32" in label:
        return 0.65

    if "base64" in label:
        return 0.55

    return 0.50




# ==============================
# Cipher Validation / Safety Override
# ==============================

def _printable_ratio(text: str) -> float:
    """
    Measures how readable decoded text is for this project.

    Important:
    - Python treats the Unicode replacement character (�) as printable.
    - Wrong Base64 decoding of a Hex string can produce many replacement
      characters and still look "printable".
    - Therefore this project uses a stricter ASCII-readable ratio.
    """

    if not text:
        return 0.0

    readable = 0
    for ch in str(text):
        code = ord(ch)
        if ch in "\r\n\t":
            readable += 1
        elif 32 <= code <= 126:
            readable += 1

    return readable / max(len(str(text)), 1)


def _replacement_ratio(text: str) -> float:
    """
    Ratio of UTF-8 replacement characters after decoding.

    A high value means the selected decoder probably decoded the wrong cipher
    family. This prevents Hex payloads from being accepted as Base64.
    """

    text = str(text or "")
    if not text:
        return 0.0

    return text.count("\ufffd") / max(len(text), 1)


def _normalize_padding(value: str, block_size: int) -> str:
    """
    Adds missing padding to Base64/Base32-like tokens.
    """

    value = value.strip()
    value = value.rstrip("=")
    value += "=" * (-len(value) % block_size)
    return value


def _try_decode_base64(token: str) -> Dict[str, Any]:
    """
    Strict Base64 validator.

    It is intentionally stricter than a simple alphabet check:
    - valid Base64 alphabet only
    - valid padding after normalization
    - successful deterministic decode
    - decoded text must be mostly printable
    """

    clean = re.sub(r"\s+", "", str(token or "").strip())

    if len(clean) < 8:
        return {"valid": False, "confidence": 0.0, "reason": "Too short for reliable Base64."}

    if not re.fullmatch(r"[A-Za-z0-9+/=]+", clean):
        return {"valid": False, "confidence": 0.0, "reason": "Invalid Base64 alphabet."}

    padded = _normalize_padding(clean, 4)

    try:
        decoded_bytes = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as e:
        return {"valid": False, "confidence": 0.0, "reason": f"Base64 decode failed: {e}"}

    if not decoded_bytes:
        return {"valid": False, "confidence": 0.0, "reason": "Decoded Base64 bytes are empty."}

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    printable = _printable_ratio(decoded_text)
    replacement = _replacement_ratio(decoded_text)

    if replacement > 0.03:
        return {
            "valid": False,
            "confidence": 0.0,
            "reason": "Base64 decoded output contains too many UTF-8 replacement characters; likely wrong cipher family.",
            "decoded_preview": decoded_text[:160],
            "replacement_ratio": round(replacement, 4),
        }

    if printable < 0.82:
        return {
            "valid": False,
            "confidence": round(printable, 4),
            "reason": "Base64 decoded output is not sufficiently printable.",
            "decoded_preview": decoded_text[:160],
        }

    # Confidence bonuses for clear Base64-only markers.
    confidence = 0.86

    if any(ch in clean for ch in "+/"):
        confidence += 0.06

    if "=" in clean:
        confidence += 0.03

    # A token containing 0/1/8/9 cannot be standard Base32, so this is strong Base64 evidence.
    if any(ch in clean for ch in "0189"):
        confidence += 0.08

    if printable >= 0.95:
        confidence += 0.03

    return {
        "valid": True,
        "cipher_type": "Base64",
        "confidence": round(min(confidence, 0.99), 4),
        "reason": "Strict Base64 validation succeeded with readable decoded text.",
        "decoded_preview": decoded_text[:160],
        "printable_ratio": round(printable, 4),
    }


def _try_decode_base32(token: str) -> Dict[str, Any]:
    """
    Strict Base32 validator.

    Base32 alphabet is A-Z and 2-7 only, plus optional padding.
    This prevents Base64 strings containing 0, 1, 8, or 9 from being
    mistakenly accepted as Base32.
    """

    clean = re.sub(r"\s+", "", str(token or "").strip()).upper()

    if len(clean) < 8:
        return {"valid": False, "confidence": 0.0, "reason": "Too short for reliable Base32."}

    if not re.fullmatch(r"[A-Z2-7=]+", clean):
        return {"valid": False, "confidence": 0.0, "reason": "Invalid Base32 alphabet."}

    padded = _normalize_padding(clean, 8)

    try:
        decoded_bytes = base64.b32decode(padded, casefold=True)
    except (binascii.Error, ValueError) as e:
        return {"valid": False, "confidence": 0.0, "reason": f"Base32 decode failed: {e}"}

    if not decoded_bytes:
        return {"valid": False, "confidence": 0.0, "reason": "Decoded Base32 bytes are empty."}

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    printable = _printable_ratio(decoded_text)
    replacement = _replacement_ratio(decoded_text)

    if replacement > 0.03:
        return {
            "valid": False,
            "confidence": 0.0,
            "reason": "Base32 decoded output contains too many UTF-8 replacement characters; likely wrong cipher family.",
            "decoded_preview": decoded_text[:160],
            "replacement_ratio": round(replacement, 4),
        }

    if printable < 0.82:
        return {
            "valid": False,
            "confidence": round(printable, 4),
            "reason": "Base32 decoded output is not sufficiently printable.",
            "decoded_preview": decoded_text[:160],
        }

    confidence = 0.84

    if "=" in clean:
        confidence += 0.04

    # Digits 2-7 are common in Base32 and not in plain uppercase text.
    if re.search(r"[2-7]", clean):
        confidence += 0.07

    if printable >= 0.95:
        confidence += 0.03

    return {
        "valid": True,
        "cipher_type": "Base32",
        "confidence": round(min(confidence, 0.98), 4),
        "reason": "Strict Base32 validation succeeded with readable decoded text.",
        "decoded_preview": decoded_text[:160],
        "printable_ratio": round(printable, 4),
    }


def _validate_morse(token: str) -> Dict[str, Any]:
    """
    Strict Morse validator based on character set and token structure.
    """

    text = str(token or "").strip()

    if len(text) < 3:
        return {"valid": False, "confidence": 0.0, "reason": "Too short for Morse."}

    if not re.fullmatch(r"[.\-\s/]+", text):
        return {"valid": False, "confidence": 0.0, "reason": "Invalid Morse alphabet."}

    symbols = [part for part in re.split(r"\s+|/", text) if part]

    if len(symbols) < 2:
        return {"valid": False, "confidence": 0.0, "reason": "Not enough Morse symbols."}

    dot_dash_count = sum(ch in ".-" for ch in text)
    symbol_ratio = dot_dash_count / max(len(text.replace(" ", "")), 1)

    confidence = 0.90

    if "/" in text:
        confidence += 0.04

    if symbol_ratio >= 0.85:
        confidence += 0.04

    return {
        "valid": True,
        "cipher_type": "Morse",
        "confidence": round(min(confidence, 0.99), 4),
        "reason": "Strict Morse validation succeeded.",
        "symbol_count": len(symbols),
        "symbol_ratio": round(symbol_ratio, 4),
    }



def _try_decode_hex(token: str) -> Dict[str, Any]:
    """
    Strict Hex / Hexadecimal validator.

    Supports common separators such as spaces, colons, hyphens, commas,
    and optional 0x prefixes. Validation requires:
    - hexadecimal characters only after cleaning
    - even number of hex digits
    - successful UTF-8 decoding
    - mostly printable decoded output
    """

    raw = str(token or "").strip()

    if not raw:
        return {"valid": False, "confidence": 0.0, "reason": "Empty token."}

    # Remove common hex formatting without accepting arbitrary prose.
    clean = re.sub(r"0x", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"[\s:\-_,]+", "", clean)

    if len(clean) < 6:
        return {"valid": False, "confidence": 0.0, "reason": "Too short for reliable Hex."}

    if len(clean) % 2 != 0:
        return {"valid": False, "confidence": 0.0, "reason": "Hex digit count is odd."}

    if not re.fullmatch(r"[0-9A-Fa-f]+", clean):
        return {"valid": False, "confidence": 0.0, "reason": "Invalid Hex alphabet."}

    try:
        decoded_bytes = bytes.fromhex(clean)
    except ValueError as e:
        return {"valid": False, "confidence": 0.0, "reason": f"Hex decode failed: {e}"}

    if not decoded_bytes:
        return {"valid": False, "confidence": 0.0, "reason": "Decoded Hex bytes are empty."}

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    printable = _printable_ratio(decoded_text)
    replacement = _replacement_ratio(decoded_text)

    if replacement > 0.03:
        return {
            "valid": False,
            "confidence": 0.0,
            "reason": "Hex decoded output contains too many UTF-8 replacement characters.",
            "decoded_preview": decoded_text[:160],
            "replacement_ratio": round(replacement, 4),
        }

    if printable < 0.82:
        return {
            "valid": False,
            "confidence": round(printable, 4),
            "reason": "Hex decoded output is not sufficiently printable.",
            "decoded_preview": decoded_text[:160],
        }

    confidence = 0.90

    if re.search(r"0x", raw, flags=re.IGNORECASE):
        confidence += 0.04

    if re.search(r"[\s:\-_,]", raw):
        confidence += 0.03

    if printable >= 0.95:
        confidence += 0.03

    return {
        "valid": True,
        "cipher_type": "Hex",
        "confidence": round(min(confidence, 0.99), 4),
        "reason": "Strict Hex validation succeeded with readable decoded text.",
        "decoded_preview": decoded_text[:160],
        "printable_ratio": round(printable, 4),
        "hex_digit_count": len(clean),
    }


def _try_decode_ascii(token: str) -> Dict[str, Any]:
    """
    Strict ASCII-decimal validator.

    Accepts decimal ASCII codes separated by spaces, commas, hyphens, pipes,
    or semicolons, for example:
        83 79 67 32 65 76 69 82 84

    The validator avoids interpreting normal Base64/Base32 strings as ASCII by
    requiring multiple numeric tokens and separator evidence.
    """

    raw = str(token or "").strip()

    if not raw:
        return {"valid": False, "confidence": 0.0, "reason": "Empty token."}

    numbers = re.findall(r"\d{1,3}", raw)

    if len(numbers) < 3:
        return {"valid": False, "confidence": 0.0, "reason": "Too few ASCII decimal tokens."}

    # Reject if the token has alphabetic characters. ASCII decimal payloads
    # should be numeric plus separators only.
    if re.search(r"[A-Za-z]", raw):
        return {"valid": False, "confidence": 0.0, "reason": "ASCII decimal text contains alphabetic characters."}

    if not re.fullmatch(r"[0-9\s,;|:/\-_.]+", raw):
        return {"valid": False, "confidence": 0.0, "reason": "Invalid ASCII decimal separator pattern."}

    try:
        values = [int(num) for num in numbers]
    except Exception as e:
        return {"valid": False, "confidence": 0.0, "reason": f"ASCII parse failed: {e}"}

    valid_values = [value for value in values if 0 <= value <= 255]

    if len(valid_values) != len(values):
        return {"valid": False, "confidence": 0.0, "reason": "One or more ASCII values are outside 0-255."}

    decoded_text = "".join(chr(value) for value in values)
    printable = _printable_ratio(decoded_text)

    if printable < 0.82:
        return {
            "valid": False,
            "confidence": round(printable, 4),
            "reason": "ASCII decoded output is not sufficiently printable.",
            "decoded_preview": decoded_text[:160],
        }

    # Strong evidence: multiple numeric tokens with separators.
    separator_count = len(re.findall(r"[\s,;|:/\-_.]+", raw))

    confidence = 0.88

    if separator_count >= 2:
        confidence += 0.05

    if all(9 <= value <= 126 or value in {10, 13} for value in values):
        confidence += 0.04

    if printable >= 0.95:
        confidence += 0.03

    return {
        "valid": True,
        "cipher_type": "ASCII",
        "confidence": round(min(confidence, 0.99), 4),
        "reason": "Strict ASCII decimal validation succeeded with readable decoded text.",
        "decoded_preview": decoded_text[:160],
        "printable_ratio": round(printable, 4),
        "ascii_token_count": len(values),
        "separator_count": separator_count,
    }

def validate_cipher_prediction(
    cypher_text: str,
    svm_prediction: str,
    svm_confidence: float,
    override_threshold: float = 0.88,
) -> Dict[str, Any]:
    """
    Deterministic safety validator for live cipher classification.

    The SVM remains the primary classifier, but deterministic validation acts as
    a guardrail for obvious cases, such as a valid Base64 command incorrectly
    predicted as Base32.
    """

    svm_prediction = str(svm_prediction or "").strip()
    svm_confidence = float(svm_confidence or 0.0)

    checks = {
        "Base64": _try_decode_base64(cypher_text),
        "Base32": _try_decode_base32(cypher_text),
        "Morse": _validate_morse(cypher_text),
        "Hex": _try_decode_hex(cypher_text),
        "ASCII": _try_decode_ascii(cypher_text),
    }

    valid_candidates = [
        result for result in checks.values()
        if result.get("valid") and result.get("cipher_type")
    ]

    if valid_candidates:
        # Collision handling:
        # A pure Hex token is also made only of Base64 alphabet characters.
        # If both validators pass, prefer the validator with stronger structural
        # evidence, not only the raw confidence score.
        raw_clean = re.sub(r"[\s:\-_,]+", "", re.sub(r"0x", "", str(cypher_text or "").strip(), flags=re.IGNORECASE))
        pure_hex_shape = (
            len(raw_clean) >= 6
            and len(raw_clean) % 2 == 0
            and re.fullmatch(r"[0-9A-Fa-f]+", raw_clean) is not None
        )

        def _candidate_priority(item: Dict[str, Any]) -> tuple[float, float]:
            cipher = str(item.get("cipher_type", ""))
            confidence = float(item.get("confidence", 0.0))
            priority_bonus = 0.0

            if pure_hex_shape and cipher == "Hex":
                priority_bonus += 0.20

            if cipher == "ASCII" and item.get("ascii_token_count", 0) >= 3:
                priority_bonus += 0.12

            if cipher == "Morse":
                priority_bonus += 0.10

            return (confidence + priority_bonus, confidence)

        best = sorted(
            valid_candidates,
            key=_candidate_priority,
            reverse=True,
        )[0]

        deterministic_type = str(best.get("cipher_type"))
        deterministic_confidence = float(best.get("confidence", 0.0))
    else:
        deterministic_type = svm_prediction
        deterministic_confidence = 0.0
        best = {
            "valid": False,
            "cipher_type": svm_prediction,
            "confidence": 0.0,
            "reason": "No deterministic validator produced a reliable cipher type.",
        }

    override_applied = (
        deterministic_type.lower() != svm_prediction.lower()
        and deterministic_confidence >= override_threshold
    )

    final_prediction = deterministic_type if override_applied else svm_prediction

    if override_applied:
        reason = (
            f"Deterministic validator overrode SVM prediction from {svm_prediction} "
            f"to {deterministic_type} because validation confidence="
            f"{deterministic_confidence:.4f} exceeded threshold={override_threshold:.4f}."
        )
    else:
        reason = (
            f"SVM prediction kept as {svm_prediction}. Best deterministic type="
            f"{deterministic_type} with confidence={deterministic_confidence:.4f}."
        )

    return {
        "success": True,
        "method": "svm_with_deterministic_cipher_validation_guardrail",
        "original_svm_prediction": svm_prediction,
        "original_svm_confidence": round(svm_confidence, 4),
        "final_prediction": final_prediction,
        "override_applied": bool(override_applied),
        "deterministic_type": deterministic_type,
        "deterministic_confidence": round(deterministic_confidence, 4),
        "override_threshold": round(override_threshold, 4),
        "reason": reason,
        "best_validator_result": best,
        "validation_scores": checks,
    }

def coerce_live_features(
    live_df: pd.DataFrame,
    training_labeled_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Ensures the live row has exactly the saved training feature columns.
    Missing values are filled using training medians, then zero fallback.
    """

    live_features = pd.DataFrame()

    for col in feature_cols:
        if col in live_df.columns:
            live_features[col] = pd.to_numeric(
                live_df[col],
                errors="coerce",
            )
        else:
            live_features[col] = np.nan

        if col in training_labeled_df.columns:
            median_value = pd.to_numeric(
                training_labeled_df[col],
                errors="coerce",
            ).median()
        else:
            median_value = 0.0

        if pd.isna(median_value):
            median_value = 0.0

        live_features[col] = live_features[col].fillna(median_value)

    return live_features


def assign_live_cluster_from_nearest_training_row(
    live_df: pd.DataFrame,
    live_scaled: np.ndarray,
    training_labeled_df: pd.DataFrame,
    scaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Assigns approximate DBSCAN/FCM cluster labels to the live row without retraining.

    Why:
    - DBSCAN does not naturally provide predict() for new samples.
    - FCM here was trained offline.
    - For live inference, we assign the nearest training row's cluster labels.
    - This avoids retraining models on every Streamlit run.
    """

    live_df = live_df.copy()

    try:
        training_features = training_labeled_df[feature_cols].copy()

        for col in feature_cols:
            training_features[col] = pd.to_numeric(
                training_features[col],
                errors="coerce",
            )

            median_value = training_features[col].median()
            if pd.isna(median_value):
                median_value = 0.0

            training_features[col] = training_features[col].fillna(median_value)

        training_scaled = scaler.transform(training_features)

        distances = np.linalg.norm(
            training_scaled - live_scaled[0],
            axis=1,
        )

        nearest_index = int(np.argmin(distances))
        nearest_row = training_labeled_df.iloc[nearest_index]

        for col in [
            "dbscan_cluster",
            "fcm_cluster",
            "fcm_membership_score",
            "risk_label",
            "business_risk_label",
        ]:
            if col in training_labeled_df.columns:
                live_df[col] = nearest_row.get(col)

        live_df["nearest_training_row_index"] = nearest_index
        live_df["nearest_training_distance"] = float(distances[nearest_index])

    except Exception:
        if "dbscan_cluster" not in live_df.columns:
            live_df["dbscan_cluster"] = "live_not_reclustered"
        if "fcm_cluster" not in live_df.columns:
            live_df["fcm_cluster"] = "live_not_reclustered"
        if "fcm_membership_score" not in live_df.columns:
            live_df["fcm_membership_score"] = 0.0

    return live_df


def prepare_live_row(extracted_cypher_text: str) -> pd.DataFrame:
    """
    Creates and feature-engineers the live input row only.
    """

    live_df = pd.DataFrame(
        [
            {
                "cypher_text": extracted_cypher_text,
                "source_cipher_type": "LiveInput",
                "source_cipher_column": "live_message",
                "source_row_id": "live",
                "is_llm_extracted": True,
                "record_date": pd.Timestamp.today().normalize(),
            }
        ]
    )

    live_df = add_cipher_features(live_df)

    return live_df


def build_forecast_dataframe(
    training_labeled_df: pd.DataFrame,
    live_df: pd.DataFrame,
    date_col: str,
    forecast_target: str,
) -> pd.DataFrame:
    """
    Builds the dataframe used by SARIMA in live mode.

    It uses saved training history + the current live row as the latest point.
    """

    forecast_df = training_labeled_df.copy()

    if date_col not in forecast_df.columns:
        forecast_df[date_col] = pd.date_range(
            start="2024-01-01",
            periods=len(forecast_df),
            freq="D",
        )

    forecast_df[date_col] = pd.to_datetime(
        forecast_df[date_col],
        errors="coerce",
    )

    if forecast_df[date_col].isna().all():
        forecast_df[date_col] = pd.date_range(
            start="2024-01-01",
            periods=len(forecast_df),
            freq="D",
        )

    if forecast_target not in forecast_df.columns:
        forecast_df[forecast_target] = 0.0

    if forecast_target not in live_df.columns:
        live_df[forecast_target] = 0.0

    max_date = pd.to_datetime(forecast_df[date_col], errors="coerce").max()

    if pd.isna(max_date):
        live_date = pd.Timestamp.today().normalize()
    else:
        live_date = max_date + pd.Timedelta(days=1)

    live_forecast_row = {
        date_col: live_date,
        forecast_target: float(pd.to_numeric(live_df[forecast_target], errors="coerce").iloc[0]),
    }

    forecast_df = pd.concat(
        [
            forecast_df[[date_col, forecast_target]],
            pd.DataFrame([live_forecast_row]),
        ],
        ignore_index=True,
    )

    return forecast_df


def calculate_action_score_safe(
    svm_confidence: float,
    shap_importance_score: float,
    forecast_risk_score: float,
    drift_score: float,
    cipher_risk_weight: float,
    fcm_membership_score: float | None = None,
    predicted_class: str = "",
    business_risk_label: str = "",
    decoded_plain_text: str = "",
    expected_change_pct: float = 0.0,
    shap_cosine_similarity: float | None = None,
    second_best_confidence: float | None = None,
) -> Dict[str, Any]:
    """
    Calculates the exact five-signal Intelligent Score.

    Required independent signals:
    1. S = dynamic class severity
    2. Gap(SVM) = classifier confidence / separation
    3. μ(predicted) = FCM membership for the predicted/live case
    4. ARIMA = forecast trend signal
    5. SHAP = prediction reliability via cosine similarity when available

    The wrapper stays backward-compatible with older versions of action_score.py.
    """

    try:
        result = calculate_total_action_score(
            svm_confidence=svm_confidence,
            shap_importance_score=shap_importance_score,
            forecast_risk_score=forecast_risk_score,
            drift_score=drift_score,
            cipher_risk_weight=cipher_risk_weight,
            fcm_membership_score=fcm_membership_score,
            predicted_class=predicted_class,
            business_risk_label=business_risk_label,
            decoded_plain_text=decoded_plain_text,
            expected_change_pct=expected_change_pct,
            shap_cosine_similarity=shap_cosine_similarity,
            second_best_confidence=second_best_confidence,
        )

        if isinstance(result, dict):
            xai_score = round(float(svm_confidence or 0.0) * float(shap_importance_score or 0.0), 4)
            result.setdefault("score_type", "exact_five_signal_intelligent_score")
            result.setdefault("xai_score", xai_score)
            result.setdefault("xai_formula", "XAI Score = SVM Confidence × SHAP Importance")
            return result

    except TypeError:
        # Older action_score.py does not accept the new arguments.
        pass
    except Exception:
        # Keep the live pipeline safe even if scoring fails.
        pass

    # Emergency five-signal fallback.
    svm_confidence = float(svm_confidence or 0.0)
    shap_importance_score = float(shap_importance_score or 0.0)
    forecast_risk_score = float(forecast_risk_score or 0.0)
    drift_score = float(drift_score or 0.0)
    cipher_risk_weight = float(cipher_risk_weight or 0.5)
    fcm_membership_score = float(fcm_membership_score or 0.5)
    shap_signal = (
        float(shap_cosine_similarity)
        if shap_cosine_similarity is not None
        else float(shap_importance_score)
    )

    xai_score = round(svm_confidence * shap_importance_score, 4)

    total_score = (
        0.22 * cipher_risk_weight
        + 0.23 * svm_confidence
        + 0.18 * fcm_membership_score
        + 0.17 * forecast_risk_score
        + 0.20 * shap_signal
    )

    return {
        "success": True,
        "score_type": "fallback_five_signal_intelligent_score",
        "formula": (
            "Intelligent Score = "
            "0.22*S + 0.23*Gap(SVM) + 0.18*μ(predicted) "
            "+ 0.17*ARIMA + 0.20*SHAP"
        ),
        "signals": {
            "S_class_severity": round(cipher_risk_weight, 4),
            "Gap_SVM": round(svm_confidence, 4),
            "Mu_predicted_FCM": round(fcm_membership_score, 4),
            "ARIMA_trend": round(forecast_risk_score, 4),
            "SHAP_reliability": round(shap_signal, 4),
        },
        "xai_score": xai_score,
        "xai_formula": "XAI Score = SVM Confidence × SHAP Importance",
        "total_action_score": round(float(total_score), 4),
        "threshold": 0.75,
        "action_required": bool(total_score >= 0.75),
        "decision": "ACTION_REQUIRED" if total_score >= 0.75 else "MONITOR_ONLY",
        "score_components": {
            "svm_confidence": round(svm_confidence, 4),
            "shap_importance_score": round(shap_importance_score, 4),
            "xai_score": xai_score,
            "forecast_risk_score": round(forecast_risk_score, 4),
            "drift_score": round(drift_score, 4),
            "dynamic_severity": round(cipher_risk_weight, 4),
            "fcm_membership_score": round(fcm_membership_score, 4),
            "shap_reliability": round(shap_signal, 4),
        },
    }


# ==============================
# Main Live SRA Pipeline
# ==============================

def run_crypto_ai_pipeline(
    messy_text: str,
    uploaded_file=None,
    execute_api_action: bool = False,
    force_api_action: bool = False,
    enable_dynamic_reclustering: bool = True,
    max_rows_per_type: int = 700,
) -> Dict[str, Any]:
    """
    Run the complete live Sense-Reason-Act pipeline.

    Important:
    - This function loads saved artifacts.
    - It does NOT retrain SVM every run.
    - To retrain, run:
      python train_local_pipeline.py
    """

    timeline = []

    def step(name: str, status: str, details: str = ""):
        timeline.append(
            {
                "step": name,
                "status": status,
                "details": details,
            }
        )

    artifact_status = get_artifact_status()

    with ls_trace(
        "CRYPTO AI Live Sense-Reason-Act Pipeline",
        "chain",
        inputs={
            "messy_text": messy_text,
            "execute_api_action": execute_api_action,
            "force_api_action": force_api_action,
            "enable_dynamic_reclustering": enable_dynamic_reclustering,
            "live_mode": True,
            "artifacts_exist": artifact_status.get("artifacts_exist"),
            "supported_cipher_types": SUPPORTED_CIPHER_TYPES,
        },
    ) as root_run:

        # ==============================
        # Step 0: Load Saved Artifacts
        # ==============================

        with ls_trace(
            "Step 0 - Load Saved Local Models",
            "chain",
            inputs={"artifact_status": artifact_status},
        ) as trace:
            artifacts = load_training_artifacts()

            scaler = artifacts["scaler"]
            svm_model = artifacts["svm_model"]
            dbscan_model = artifacts.get("dbscan_model")
            fcm_info = artifacts.get("fcm_info", {})
            feature_cols = artifacts["feature_cols"]
            training_labeled_df = artifacts["labeled_df"]
            shap_background_df = artifacts["shap_background_df"]
            metadata = artifacts["metadata"]

            end_trace(
                trace,
                {
                    "loaded": True,
                    "feature_count": len(feature_cols),
                    "training_rows": len(training_labeled_df),
                    "metadata": metadata,
                    "paths": artifacts.get("paths", {}),
                },
            )

        step(
            "Load Saved Local Models",
            "success",
            (
                "Loaded saved scaler and SVM-RBF model from artifacts/. "
                "SVM was NOT retrained during this Streamlit run."
            ),
        )

        # ==============================
        # Step 1: Optional Data Preview / Raw Dataset
        # ==============================

        with ls_trace(
            "Step 1 - Automated Data Ingestion / Preview",
            "chain",
            inputs={"uploaded_file_provided": uploaded_file is not None},
        ) as trace:
            try:
                raw_df = load_data(uploaded_file)
            except Exception:
                raw_df = training_labeled_df.copy()

            working_df = raw_df.copy()

            end_trace(
                trace,
                {
                    "raw_shape": list(raw_df.shape),
                    "working_shape": list(working_df.shape),
                    "columns_sample": list(raw_df.columns[:20]),
                },
            )

        step(
            "Automated Data Ingestion",
            "success",
            f"Loaded preview/raw data shape={raw_df.shape}. Live inference uses saved artifacts.",
        )

        # ==============================
        # Step 2: LLM / Regex Extraction
        # ==============================

        with ls_trace(
            "Step 2 - LLM/Regex Cypher Text Extraction",
            "chain",
            inputs={"messy_text": messy_text},
        ) as trace:
            extraction_result = extract_cypher_text(messy_text)

            extracted_cypher_text = str(
                extraction_result.get("cypher_text", "")
            ).strip()

            end_trace(
                trace,
                {
                    "source": extraction_result.get("source"),
                    "cypher_text": extracted_cypher_text,
                    "confidence": extraction_result.get("confidence"),
                    "reason": extraction_result.get("reason"),
                },
            )

        step(
            "LLM Cypher Text Extraction",
            "success" if extracted_cypher_text else "warning",
            extraction_result.get("reason", ""),
        )

        if not extracted_cypher_text:
            raise ValueError(
                "No cypher_text was extracted. Please enter a supported Base64, Base32, Morse, Hex, or ASCII message."
            )

        # ==============================
        # Step 3: Live Feature Engineering
        # ==============================

        with ls_trace(
            "Step 3 - Live Feature Engineering",
            "chain",
            inputs={"extracted_cypher_text": extracted_cypher_text},
        ) as trace:
            live_df = prepare_live_row(extracted_cypher_text)

            live_features = coerce_live_features(
                live_df=live_df,
                training_labeled_df=training_labeled_df,
                feature_cols=feature_cols,
            )

            live_scaled = scaler.transform(live_features)

            end_trace(
                trace,
                {
                    "live_df_shape": list(live_df.shape),
                    "feature_cols": feature_cols,
                    "live_scaled_shape": list(live_scaled.shape),
                },
            )

        step(
            "Live Feature Engineering",
            "success",
            f"Generated live features and scaled them using saved scaler. Shape={live_scaled.shape}",
        )

        # ==============================
        # Step 4: DBSCAN/FCM Live Assignment
        # ==============================

        with ls_trace(
            "Step 4 - DBSCAN/FCM Live Cluster Assignment",
            "chain",
            inputs={
                "method": "nearest_saved_training_row_no_retraining",
                "live_scaled_shape": list(live_scaled.shape),
            },
        ) as trace:
            live_df = assign_live_cluster_from_nearest_training_row(
                live_df=live_df,
                live_scaled=live_scaled,
                training_labeled_df=training_labeled_df,
                scaler=scaler,
                feature_cols=feature_cols,
            )

            end_trace(
                trace,
                {
                    "dbscan_cluster": live_df.get("dbscan_cluster", pd.Series([""])).iloc[0],
                    "fcm_cluster": live_df.get("fcm_cluster", pd.Series([""])).iloc[0],
                    "fcm_membership_score": live_df.get(
                        "fcm_membership_score",
                        pd.Series([0.0]),
                    ).iloc[0],
                    "nearest_training_row_index": live_df.get(
                        "nearest_training_row_index",
                        pd.Series([None]),
                    ).iloc[0],
                    "nearest_training_distance": live_df.get(
                        "nearest_training_distance",
                        pd.Series([None]),
                    ).iloc[0],
                },
            )

        step(
            "DBSCAN/FCM Live Assignment",
            "success",
            (
                "Assigned live DBSCAN/FCM context using nearest saved training row. "
                "No clustering retraining was performed."
            ),
        )

        # ==============================
        # Step 5: Saved SVM-RBF Prediction
        # ==============================

        with ls_trace(
            "Step 5 - Saved SVM-RBF Live Classification",
            "chain",
            inputs={
                "live_scaled_shape": list(live_scaled.shape),
                "feature_count": len(feature_cols),
            },
        ) as trace:
            predictions, confidence_scores = predict_with_confidence(
                svm_model,
                live_scaled,
            )

            extracted_prediction = str(predictions[0])
            extracted_confidence = float(confidence_scores[0])

            live_df["svm_predicted_cipher_type"] = extracted_prediction
            live_df["svm_prediction"] = extracted_prediction
            live_df["svm_confidence"] = extracted_confidence

            end_trace(
                trace,
                {
                    "svm_model": svm_model.__class__.__name__,
                    "svm_kernel": getattr(svm_model, "kernel", "rbf"),
                    "prediction": extracted_prediction,
                    "confidence": extracted_confidence,
                },
            )

        step(
            "SVM-RBF Live Classification",
            "success",
            (
                f"Saved SVM predicted {extracted_prediction} "
                f"with confidence={round(extracted_confidence, 4)}. "
                "SVM was NOT trained on this run."
            ),
        )

        # ==============================
        # Step 5.1: Deterministic Cipher Validation Guardrail
        # ==============================

        with ls_trace(
            "Step 5.1 - Deterministic Cipher Validation Guardrail",
            "chain",
            inputs={
                "cypher_text": extracted_cypher_text,
                "svm_prediction": extracted_prediction,
                "svm_confidence": extracted_confidence,
            },
        ) as trace:
            svm_original_prediction = extracted_prediction

            cipher_validation_result = validate_cipher_prediction(
                cypher_text=extracted_cypher_text,
                svm_prediction=extracted_prediction,
                svm_confidence=extracted_confidence,
                override_threshold=0.88,
            )

            extracted_prediction = str(
                cipher_validation_result.get(
                    "final_prediction",
                    extracted_prediction,
                )
            )

            # Keep both the raw SVM prediction and the validated final prediction.
            live_df["svm_original_prediction"] = svm_original_prediction
            live_df["svm_predicted_cipher_type"] = extracted_prediction
            live_df["svm_prediction"] = extracted_prediction
            live_df["validated_cipher_type"] = extracted_prediction
            live_df["cipher_validation_override"] = bool(
                cipher_validation_result.get("override_applied", False)
            )
            live_df["cipher_validation_confidence"] = float(
                cipher_validation_result.get("deterministic_confidence", 0.0)
            )

            end_trace(
                trace,
                {
                    "original_svm_prediction": svm_original_prediction,
                    "final_prediction": extracted_prediction,
                    "override_applied": cipher_validation_result.get("override_applied"),
                    "deterministic_type": cipher_validation_result.get("deterministic_type"),
                    "deterministic_confidence": cipher_validation_result.get("deterministic_confidence"),
                    "reason": cipher_validation_result.get("reason"),
                    "validation_scores": cipher_validation_result.get("validation_scores"),
                },
            )

        step(
            "Deterministic Cipher Validation Guardrail",
            "warning" if cipher_validation_result.get("override_applied") else "success",
            (
                f"SVM={cipher_validation_result.get('original_svm_prediction')} → "
                f"Final={cipher_validation_result.get('final_prediction')}; "
                f"override={cipher_validation_result.get('override_applied')}; "
                f"deterministic_confidence="
                f"{cipher_validation_result.get('deterministic_confidence')}; "
                f"reason={cipher_validation_result.get('reason')}"
            ),
        )

        # ==============================
        # Step 6: Decode After SVM
        # ==============================

        with ls_trace(
            "Step 6 - Deterministic Decode After SVM",
            "tool",
            inputs={
                "cypher_text": extracted_cypher_text,
                "svm_predicted_cipher_type": extracted_prediction,
            },
        ) as trace:
            decode_result = decode_by_type(
                extracted_cypher_text,
                extracted_prediction,
            )

            decoded_plain_text = (
                decode_result.get("decoded_text", "")
                if decode_result.get("success")
                else ""
            )

            end_trace(
                trace,
                {
                    "decode_success": bool(decode_result.get("success")),
                    "decoded_text": decoded_plain_text,
                    "message": decode_result.get("message", ""),
                },
            )

        step(
            "Deterministic Decode After SVM",
            "success" if decode_result.get("success") else "warning",
            decode_result.get("message", ""),
        )

        # ==============================
        # Step 7: Mandatory RAG Query Preparation
        # ==============================

        with ls_trace(
            "Step 7 - Mandatory RAG Query Preparation",
            "chain",
            inputs={
                "prediction": extracted_prediction,
                "cypher_text": extracted_cypher_text,
                "decoded_plain_text": decoded_plain_text,
                "svm_confidence": extracted_confidence,
            },
        ) as trace:
            try:
                rag_query = build_cipher_rag_query(
                    extracted_prediction,
                    extracted_cypher_text,
                    decoded_plain_text,
                    extracted_confidence,
                )

                rag_result = {
                    "query": rag_query,
                    "retrieved_context": "",
                    "retrieved_items": [],
                    "retrieval_executed": False,
                    "retrieval_policy": (
                        "Explicit RAG retrieval is a mandatory decision layer, "
                        "but actual retrieval is executed only when the live "
                        "Intelligent Score is borderline."
                    ),
                }

            except Exception as e:
                rag_query = ""
                rag_result = {
                    "query": "",
                    "retrieved_context": "",
                    "retrieved_items": [],
                    "retrieval_executed": False,
                    "error": str(e),
                }

            end_trace(
                trace,
                {
                    "query": rag_query,
                    "retrieval_executed": False,
                    "policy": rag_result.get("retrieval_policy"),
                    "error": rag_result.get("error"),
                },
            )

        step(
            "Mandatory RAG Query Preparation",
            "success" if rag_query else "warning",
            (
                "Prepared RAG query. Retrieval is not optional, but it is "
                "executed later only if the case is borderline, as required."
            ),
        )

        # ==============================
        # Step 8: SHAP Explainability
        # ==============================

        with ls_trace(
            "Step 8 - SHAP Explainability",
            "chain",
            inputs={
                "prediction": extracted_prediction,
                "live_scaled_shape": list(live_scaled.shape),
                "feature_cols": feature_cols,
            },
        ) as trace:
            try:
                shap_background_features = coerce_live_features(
                    live_df=shap_background_df,
                    training_labeled_df=training_labeled_df,
                    feature_cols=feature_cols,
                )

                shap_background_scaled = scaler.transform(shap_background_features)

                # Live row first; max_explain_samples=1 explains only the live row.
                X_for_shap = np.vstack(
                    [
                        live_scaled,
                        shap_background_scaled,
                    ]
                )

                shap_result = run_shap_explanation(
                    svm_model,
                    X_for_shap,
                    feature_cols,
                    max_background_samples=min(50, len(X_for_shap)),
                    max_explain_samples=1,
                )

            except Exception as e:
                shap_result = {
                    "shap_score": 0.5,
                    "importance_df": pd.DataFrame(
                        {
                            "feature": feature_cols,
                            "shap_importance": [0.0] * len(feature_cols),
                        }
                    ),
                    "engine": "fallback",
                    "error": str(e),
                }

            shap_score = float(shap_result.get("shap_score", 0.5))
            importance_df = shap_result.get("importance_df", pd.DataFrame())

            top_importance = []
            if importance_df is not None and not importance_df.empty:
                top_importance = importance_df.head(10).to_dict(orient="records")

            end_trace(
                trace,
                {
                    "shap_engine": shap_result.get("engine"),
                    "shap_score": shap_score,
                    "top_importance": top_importance,
                    "error": shap_result.get("error"),
                },
            )

        step(
            "SHAP Explainability",
            "success",
            f"SHAP score={round(shap_score, 4)}; engine={shap_result.get('engine', 'unknown')}",
        )

        # ==============================
        # Step 8.1: SHAP Cluster Mean Cosine Reliability
        # ==============================

        with ls_trace(
            "Step 8.1 - SHAP Cluster Mean Cosine Reliability",
            "chain",
            inputs={
                "prediction": extracted_prediction,
                "feature_count": len(feature_cols),
                "live_df_shape": list(live_df.shape),
                "training_rows": len(training_labeled_df),
            },
        ) as trace:
            try:
                shap_reliability_result = calculate_shap_cluster_reliability(
                    importance_df=importance_df,
                    feature_cols=feature_cols,
                    training_labeled_df=training_labeled_df,
                    live_df=live_df,
                    predicted_class=extracted_prediction,
                )

                shap_cosine_similarity = float(
                    shap_reliability_result.get(
                        "shap_cosine_similarity",
                        shap_score,
                    )
                )

            except Exception as e:
                shap_cosine_similarity = float(shap_score)

                shap_reliability_result = {
                    "success": False,
                    "method": "fallback_to_shap_importance",
                    "shap_cosine_similarity": shap_cosine_similarity,
                    "shap_reliability": shap_cosine_similarity,
                    "reliability_level": "fallback",
                    "error": str(e),
                }

            # Make the reliability signal available to downstream scoring, reporting,
            # Streamlit display, and LangSmith tracing.
            shap_result["shap_cosine_similarity"] = shap_cosine_similarity
            shap_result["shap_reliability_result"] = shap_reliability_result

            end_trace(
                trace,
                {
                    "method": shap_reliability_result.get("method"),
                    "shap_cosine_similarity": shap_cosine_similarity,
                    "reliability_level": shap_reliability_result.get("reliability_level"),
                    "cluster_column": shap_reliability_result.get("cluster_column"),
                    "cluster_value": shap_reliability_result.get("cluster_value"),
                    "matched_rows": shap_reliability_result.get("matched_rows"),
                    "fallback_used": shap_reliability_result.get("fallback_used"),
                    "error": shap_reliability_result.get("error"),
                },
            )

        step(
            "SHAP Cluster Mean Cosine Reliability",
            "success" if shap_reliability_result.get("success") else "warning",
            (
                f"SHAP cosine similarity={round(float(shap_cosine_similarity), 4)}; "
                f"level={shap_reliability_result.get('reliability_level')}; "
                f"cluster={shap_reliability_result.get('cluster_column')}="
                f"{shap_reliability_result.get('cluster_value')}"
            ),
        )

        # ==============================
        # Step 8.2: Formal SHAP Re-check Loop
        # ==============================

        shap_recheck_result = {
            "success": True,
            "method": "shap_recheck_loop",
            "recheck_triggered": False,
            "reason": "SHAP cosine reliability is acceptable; no re-check required.",
            "original_shap_cosine_similarity": round(float(shap_cosine_similarity), 4),
            "final_shap_cosine_similarity": round(float(shap_cosine_similarity), 4),
            "final_reliability_level": shap_reliability_result.get("reliability_level", "unknown"),
            "route_after_recheck": "continue_current_route",
        }

        with ls_trace(
            "Step 8.2 - Formal SHAP Re-check Loop",
            "chain",
            inputs={
                "original_shap_cosine_similarity": shap_cosine_similarity,
                "reliability_threshold": 0.65,
                "prediction": extracted_prediction,
                "feature_count": len(feature_cols),
            },
        ) as trace:
            try:
                # Rebuild SHAP background safely for the re-check loop.
                shap_background_features_recheck = coerce_live_features(
                    live_df=shap_background_df,
                    training_labeled_df=training_labeled_df,
                    feature_cols=feature_cols,
                )

                shap_background_scaled_recheck = scaler.transform(
                    shap_background_features_recheck
                )

                shap_recheck_result = run_shap_recheck_loop(
                    svm_model=svm_model,
                    live_scaled=live_scaled,
                    shap_background_scaled=shap_background_scaled_recheck,
                    feature_cols=feature_cols,
                    training_labeled_df=training_labeled_df,
                    live_df=live_df,
                    predicted_class=extracted_prediction,
                    original_shap_result=shap_result,
                    original_shap_cosine_similarity=shap_cosine_similarity,
                    reliability_threshold=0.65,
                    max_background_samples=min(120, len(shap_background_scaled_recheck) + 1),
                )

                if shap_recheck_result.get("recheck_triggered"):
                    # Use the re-checked SHAP output downstream so Dynamic Severity,
                    # Intelligent Score, RAG, ReAct, and reporting all see the
                    # updated explanation reliability.
                    updated_shap_result = shap_recheck_result.get(
                        "updated_shap_result",
                        shap_result,
                    )

                    updated_reliability_result = shap_recheck_result.get(
                        "updated_shap_reliability_result",
                        shap_reliability_result,
                    )

                    if isinstance(updated_shap_result, dict):
                        shap_result = updated_shap_result

                    if isinstance(updated_reliability_result, dict):
                        shap_reliability_result = updated_reliability_result

                    shap_cosine_similarity = float(
                        shap_recheck_result.get(
                            "final_shap_cosine_similarity",
                            shap_cosine_similarity,
                        )
                    )

                    shap_score = float(shap_result.get("shap_score", shap_score))
                    importance_df = shap_result.get("importance_df", importance_df)

                    shap_result["shap_cosine_similarity"] = shap_cosine_similarity
                    shap_result["shap_reliability_result"] = shap_reliability_result

            except Exception as e:
                shap_recheck_result = {
                    "success": False,
                    "method": "shap_recheck_loop",
                    "recheck_triggered": True,
                    "reason": "SHAP re-check failed; downstream ReAct should route safely.",
                    "error": str(e),
                    "original_shap_cosine_similarity": round(float(shap_cosine_similarity), 4),
                    "final_shap_cosine_similarity": round(float(shap_cosine_similarity), 4),
                    "final_reliability_level": shap_reliability_result.get("reliability_level", "fallback"),
                    "route_after_recheck": "human_review_after_recheck_error",
                }

            end_trace(
                trace,
                {
                    "success": shap_recheck_result.get("success"),
                    "recheck_triggered": shap_recheck_result.get("recheck_triggered"),
                    "original_shap_cosine_similarity": shap_recheck_result.get("original_shap_cosine_similarity"),
                    "final_shap_cosine_similarity": shap_recheck_result.get("final_shap_cosine_similarity"),
                    "final_reliability_level": shap_recheck_result.get("final_reliability_level"),
                    "route_after_recheck": shap_recheck_result.get("route_after_recheck"),
                    "reason": shap_recheck_result.get("reason"),
                    "error": shap_recheck_result.get("error"),
                },
            )

        step(
            "Formal SHAP Re-check Loop",
            "warning"
            if shap_recheck_result.get("route_after_recheck") in {
                "human_review_after_failed_shap_recheck",
                "human_review_after_recheck_error",
            }
            else "success",
            (
                f"triggered={shap_recheck_result.get('recheck_triggered')}; "
                f"original_cosine={shap_recheck_result.get('original_shap_cosine_similarity')}; "
                f"final_cosine={shap_recheck_result.get('final_shap_cosine_similarity')}; "
                f"level={shap_recheck_result.get('final_reliability_level')}; "
                f"route_after_recheck={shap_recheck_result.get('route_after_recheck')}"
            ),
        )

        # ==============================
        # Step 9: SARIMA Forecast + Drift
        # ==============================

        date_col = metadata.get("date_col", "record_date")
        forecast_target = metadata.get("forecast_target", "cypher_entropy")
        risk_feature = metadata.get("risk_feature", "cypher_entropy")

        with ls_trace(
            "Step 9 - SARIMA Forecasting",
            "chain",
            inputs={
                "date_col": date_col,
                "forecast_target": forecast_target,
            },
        ) as trace:
            try:
                forecast_input_df = build_forecast_dataframe(
                    training_labeled_df=training_labeled_df,
                    live_df=live_df,
                    date_col=date_col,
                    forecast_target=forecast_target,
                )

                sarima_result = run_sarima_forecast(
                    forecast_input_df,
                    date_col=date_col,
                    target_col=forecast_target,
                )

            except Exception as e:
                sarima_result = {
                    "forecast_risk_score": 0.5,
                    "expected_change_pct": 0.0,
                    "forecast_df": pd.DataFrame(),
                    "engine": "fallback",
                    "error": str(e),
                    "last_actual_value": None,
                    "final_forecast_value": None,
                }

            forecast_risk_score = float(sarima_result.get("forecast_risk_score", 0.5))

            end_trace(
                trace,
                {
                    "forecast_risk_score": forecast_risk_score,
                    "expected_change_pct": sarima_result.get("expected_change_pct"),
                    "engine": sarima_result.get("engine"),
                    "error": sarima_result.get("error"),
                },
            )

        step(
            "SARIMA Forecasting",
            "success",
            f"Forecast risk score={round(forecast_risk_score, 4)}; engine={sarima_result.get('engine', 'unknown')}",
        )

        with ls_trace(
            "Step 10 - Drift Detection",
            "chain",
            inputs={
                "forecast_risk_score": forecast_risk_score,
                "expected_change_pct": sarima_result.get("expected_change_pct", 0.0),
            },
        ) as trace:
            drift_result = detect_drift(sarima_result)

            end_trace(
                trace,
                {
                    "drift_detected": bool(drift_result.get("drift_detected")),
                    "drift_score": drift_result.get("drift_score"),
                    "recommendation": drift_result.get("recommendation"),
                },
            )

        step(
            "Drift Detection",
            "warning" if drift_result.get("drift_detected") else "success",
            drift_result.get("recommendation", ""),
        )

        # ==============================
        # Step 10.1: Dual Drift Detection
        # ==============================

        training_scaled_for_drift = None

        with ls_trace(
            "Step 10.1 - Dual Drift Detection",
            "chain",
            inputs={
                "current_svm_confidence": extracted_confidence,
                "legacy_drift_detected": bool(drift_result.get("drift_detected")),
                "legacy_drift_score": drift_result.get("drift_score"),
                "feature_count": len(feature_cols),
            },
        ) as trace:
            try:
                training_scaled_for_drift = scaler.transform(
                    coerce_live_features(
                        live_df=training_labeled_df,
                        training_labeled_df=training_labeled_df,
                        feature_cols=feature_cols,
                    )
                )

                dual_drift_result = detect_dual_drift(
                    current_svm_confidence=extracted_confidence,
                    live_scaled=live_scaled,
                    training_scaled=training_scaled_for_drift,
                    training_labeled_df=training_labeled_df,
                    live_df=live_df,
                    existing_forecast_drift_result=drift_result,
                )

            except Exception as e:
                dual_drift_result = {
                    "success": False,
                    "method": "dual_drift_detector",
                    "dual_drift_detected": False,
                    "outer_loop_required": False,
                    "decision": "DUAL_DRIFT_EVALUATION_ERROR",
                    "recommendation": (
                        "Dual drift detection failed; do not trigger reclustering "
                        "without the two required independent signals."
                    ),
                    "signals": {},
                    "single_signal_noise": False,
                    "error": str(e),
                }

            gap_signal = dual_drift_result.get("signals", {}).get("gap_svm_trend_decline", {})
            centroid_signal = dual_drift_result.get("signals", {}).get("centroid_movement", {})

            # Attach strict dual-drift decision to the legacy drift result so later
            # dashboard/report layers can see both the old forecast signal and the
            # strict two-signal decision.
            drift_result["dual_drift_detected"] = bool(dual_drift_result.get("dual_drift_detected"))
            drift_result["outer_loop_required"] = bool(dual_drift_result.get("outer_loop_required"))
            drift_result["dual_drift_decision"] = dual_drift_result.get("decision")
            drift_result["dual_drift_recommendation"] = dual_drift_result.get("recommendation")

            end_trace(
                trace,
                {
                    "dual_drift_detected": dual_drift_result.get("dual_drift_detected"),
                    "outer_loop_required": dual_drift_result.get("outer_loop_required"),
                    "decision": dual_drift_result.get("decision"),
                    "gap_svm_decline": gap_signal.get("detected"),
                    "gap_relative_drop": gap_signal.get("relative_drop"),
                    "centroid_movement": centroid_signal.get("detected"),
                    "centroid_z_score": centroid_signal.get("z_score"),
                    "single_signal_noise": dual_drift_result.get("single_signal_noise"),
                    "recommendation": dual_drift_result.get("recommendation"),
                    "error": dual_drift_result.get("error"),
                },
            )

        step(
            "Dual Drift Detection",
            "warning" if dual_drift_result.get("dual_drift_detected") else "success",
            (
                f"dual_drift={dual_drift_result.get('dual_drift_detected')}; "
                f"gap_svm_decline={gap_signal.get('detected')}; "
                f"centroid_movement={centroid_signal.get('detected')}; "
                f"decision={dual_drift_result.get('decision')}"
            ),
        )

        # ==============================
        # Step 11: Outer ReAct Rebuild Loop
        # ==============================

        outer_react_loop_result = {
            "success": False,
            "method": "outer_react_rebuild_loop",
            "outer_loop_triggered": False,
            "decision": "OUTER_LOOP_NOT_EVALUATED",
            "status": "Outer ReAct rebuild loop has not been evaluated yet.",
            "accepted_candidate": False,
            "rollback_required": False,
        }

        dynamic_reclustering_result = None

        with ls_trace(
            "Step 11 - Outer ReAct Rebuild Loop",
            "chain",
            inputs={
                "legacy_drift_detected": bool(drift_result.get("drift_detected")),
                "dual_drift_detected": bool(dual_drift_result.get("dual_drift_detected")),
                "outer_loop_required": bool(dual_drift_result.get("outer_loop_required")),
                "enable_dynamic_reclustering": enable_dynamic_reclustering,
                "current_svm_accuracy": metadata.get("svm_accuracy"),
            },
        ) as trace:
            try:
                if training_scaled_for_drift is None:
                    training_scaled_for_drift = scaler.transform(
                        coerce_live_features(
                            live_df=training_labeled_df,
                            training_labeled_df=training_labeled_df,
                            feature_cols=feature_cols,
                        )
                    )

                outer_react_loop_result = run_outer_react_rebuild_loop(
                    training_labeled_df=training_labeled_df,
                    training_scaled=training_scaled_for_drift,
                    feature_cols=feature_cols,
                    risk_feature=risk_feature,
                    dual_drift_result=dual_drift_result,
                    current_metadata=metadata,
                    current_svm_accuracy=metadata.get("svm_accuracy", 0.0),
                    enable_rebuild=enable_dynamic_reclustering,
                    min_classifier_accuracy=0.70,
                    max_allowed_accuracy_drop=0.10,
                )

                if outer_react_loop_result.get("accepted_candidate"):
                    dynamic_reclustering_result = outer_react_loop_result.get(
                        "candidate_rebuild_result",
                        {},
                    )

                    if isinstance(dynamic_reclustering_result, dict):
                        dynamic_reclustering_result.setdefault(
                            "status",
                            outer_react_loop_result.get("status"),
                        )
                        dynamic_reclustering_result["outer_loop_accepted"] = True
                        dynamic_reclustering_result["rollback_required"] = False

                else:
                    dynamic_reclustering_result = {
                        "status": outer_react_loop_result.get("status"),
                        "triggered": bool(outer_react_loop_result.get("outer_loop_triggered")),
                        "outer_loop_accepted": False,
                        "rollback_required": bool(outer_react_loop_result.get("rollback_required")),
                        "rollback_reason": outer_react_loop_result.get("rollback_reason"),
                        "decision": outer_react_loop_result.get("decision"),
                    }

                # Attach the outer-loop decision to drift_result for downstream
                # dashboard/report layers.
                drift_result["outer_react_loop_triggered"] = bool(
                    outer_react_loop_result.get("outer_loop_triggered")
                )
                drift_result["outer_react_loop_decision"] = outer_react_loop_result.get("decision")
                drift_result["outer_react_loop_status"] = outer_react_loop_result.get("status")
                drift_result["rollback_required"] = bool(
                    outer_react_loop_result.get("rollback_required")
                )
                drift_result["outer_loop_accepted_candidate"] = bool(
                    outer_react_loop_result.get("accepted_candidate")
                )

                end_trace(
                    trace,
                    {
                        "outer_loop_triggered": outer_react_loop_result.get("outer_loop_triggered"),
                        "decision": outer_react_loop_result.get("decision"),
                        "accepted_candidate": outer_react_loop_result.get("accepted_candidate"),
                        "rollback_required": outer_react_loop_result.get("rollback_required"),
                        "current_accuracy": outer_react_loop_result.get("current_accuracy"),
                        "new_accuracy": outer_react_loop_result.get("new_accuracy"),
                        "quality_gate_passed": outer_react_loop_result.get("quality_gate_passed"),
                        "rebuild_actions": outer_react_loop_result.get("rebuild_actions"),
                        "audit_path": outer_react_loop_result.get("audit_path"),
                        "error": outer_react_loop_result.get("error"),
                    },
                )

            except Exception as e:
                outer_react_loop_result = {
                    "success": False,
                    "method": "outer_react_rebuild_loop",
                    "outer_loop_triggered": bool(dual_drift_result.get("outer_loop_required")),
                    "decision": "OUTER_LOOP_ERROR_ROLLBACK",
                    "status": "Outer ReAct rebuild loop failed; previous artifacts are preserved.",
                    "accepted_candidate": False,
                    "rollback_required": True,
                    "rollback_reason": str(e),
                    "error": str(e),
                }

                dynamic_reclustering_result = {
                    "status": "Outer ReAct loop failed; rollback to previous artifacts.",
                    "triggered": bool(dual_drift_result.get("outer_loop_required")),
                    "outer_loop_accepted": False,
                    "rollback_required": True,
                    "rollback_reason": str(e),
                    "decision": "OUTER_LOOP_ERROR_ROLLBACK",
                }

                end_trace(
                    trace,
                    {
                        "outer_loop_triggered": outer_react_loop_result.get("outer_loop_triggered"),
                        "decision": outer_react_loop_result.get("decision"),
                        "rollback_required": True,
                        "error": str(e),
                    },
                )

        step(
            "Outer ReAct Rebuild Loop",
            "warning" if outer_react_loop_result.get("rollback_required") else "success",
            (
                f"triggered={outer_react_loop_result.get('outer_loop_triggered')}; "
                f"decision={outer_react_loop_result.get('decision')}; "
                f"accepted={outer_react_loop_result.get('accepted_candidate')}; "
                f"rollback={outer_react_loop_result.get('rollback_required')}; "
                f"actions={outer_react_loop_result.get('rebuild_actions')}"
            ),
        )

        # ==============================
        # Step 12: Dynamic Severity + Exact Five-Signal Intelligent Score
        # ==============================

        with ls_trace(
            "Step 12 - Dynamic Severity + Exact Five-Signal Intelligent Score",
            "chain",
            inputs={
                "svm_confidence": extracted_confidence,
                "shap_importance_score": shap_score,
                "forecast_risk_score": forecast_risk_score,
                "drift_score": float(drift_result.get("drift_score", 0.0)),
                "cipher_label": extracted_prediction,
                "shap_cosine_similarity": shap_cosine_similarity,
            },
        ) as trace:
            avg_svm_confidence = (
                float(training_labeled_df["svm_confidence"].mean())
                if "svm_confidence" in training_labeled_df.columns
                else extracted_confidence
            )

            # ------------------------------
            # μ(predicted): FCM membership
            # ------------------------------
            live_fcm_membership_score = None

            if "fcm_membership_score" in live_df.columns:
                try:
                    live_fcm_membership_score = float(
                        pd.to_numeric(
                            live_df["fcm_membership_score"],
                            errors="coerce",
                        ).iloc[0]
                    )
                except Exception:
                    live_fcm_membership_score = None

            if live_fcm_membership_score is None:
                live_fcm_membership_score = 0.50

            # ------------------------------
            # Business risk label/context
            # ------------------------------
            live_business_risk_label = ""

            if "business_risk_label" in live_df.columns:
                try:
                    live_business_risk_label = str(live_df["business_risk_label"].iloc[0])
                except Exception:
                    live_business_risk_label = ""

            elif "risk_label" in live_df.columns:
                try:
                    live_business_risk_label = str(live_df["risk_label"].iloc[0])
                except Exception:
                    live_business_risk_label = ""

            # ------------------------------
            # Dominant SHAP feature
            # ------------------------------
            dominant_shap_feature = ""

            try:
                if importance_df is not None and not importance_df.empty:
                    sorted_importance = importance_df.sort_values(
                        "shap_importance",
                        ascending=False,
                    )
                    dominant_shap_feature = str(sorted_importance.iloc[0].get("feature", ""))
            except Exception:
                dominant_shap_feature = ""

            # ------------------------------
            # ARIMA/SARIMA trend input
            # ------------------------------
            try:
                expected_change_pct = float(sarima_result.get("expected_change_pct", 0.0))
            except Exception:
                expected_change_pct = 0.0

            drift_score_value = float(drift_result.get("drift_score", 0.0))

            # ------------------------------
            # Dynamic severity S
            # ------------------------------
            try:
                dynamic_severity_result = calculate_dynamic_severity(
                    predicted_class=extracted_prediction,
                    business_risk_label=live_business_risk_label,
                    decoded_plain_text=decoded_plain_text,
                    dominant_shap_feature=dominant_shap_feature,
                    svm_confidence=extracted_confidence,
                    fcm_membership_score=live_fcm_membership_score,
                    shap_cosine_similarity=shap_cosine_similarity,
                    forecast_risk_score=forecast_risk_score,
                    expected_change_pct=expected_change_pct,
                    drift_score=drift_score_value,
                )

                dynamic_severity_score = float(
                    dynamic_severity_result.get("dynamic_severity", 0.50)
                )

            except Exception as e:
                dynamic_severity_score = risk_weight_from_label(extracted_prediction)
                dynamic_severity_result = {
                    "success": False,
                    "method": "fallback_to_cipher_risk_weight",
                    "dynamic_severity": dynamic_severity_score,
                    "severity_level": "FALLBACK",
                    "reason_codes": [
                        "Dynamic severity failed; used cipher risk weight fallback."
                    ],
                    "error": str(e),
                }

            # ------------------------------
            # Exact five-signal intelligent score
            # ------------------------------
            action_score_result = calculate_action_score_safe(
                svm_confidence=extracted_confidence,
                shap_importance_score=shap_score,
                forecast_risk_score=forecast_risk_score,
                drift_score=drift_score_value,
                cipher_risk_weight=dynamic_severity_score,
                fcm_membership_score=live_fcm_membership_score,
                predicted_class=extracted_prediction,
                business_risk_label=live_business_risk_label,
                decoded_plain_text=decoded_plain_text,
                expected_change_pct=expected_change_pct,
                shap_cosine_similarity=shap_cosine_similarity,
                second_best_confidence=None,
            )

            total_score = float(action_score_result["total_action_score"])
            action_required = bool(action_score_result["action_required"])
            xai_score = float(action_score_result["xai_score"])
            score_signals = action_score_result.get("signals", {})

            end_trace(
                trace,
                {
                    "dynamic_severity": dynamic_severity_score,
                    "dynamic_severity_level": dynamic_severity_result.get("severity_level"),
                    "dynamic_severity_method": dynamic_severity_result.get("method"),
                    "dynamic_severity_reasons": dynamic_severity_result.get("reason_codes", []),
                    "score_type": action_score_result.get("score_type"),
                    "formula": action_score_result.get("formula"),
                    "xai_formula": action_score_result.get("xai_formula"),
                    "signals": score_signals,
                    "xai_score": xai_score,
                    "total_action_score": total_score,
                    "action_required": action_required,
                    "decision": action_score_result.get("decision"),
                    "fcm_membership_score_used": live_fcm_membership_score,
                    "business_risk_label_used": live_business_risk_label,
                    "dominant_shap_feature_used": dominant_shap_feature,
                    "expected_change_pct_used": expected_change_pct,
                    "shap_cosine_similarity_used": shap_cosine_similarity,
                },
            )

        step(
            "Dynamic Severity + Exact Five-Signal Intelligent Score",
            "success" if action_required else "info",
            (
                f"S(dynamic)={round(dynamic_severity_score, 4)} "
                f"level={dynamic_severity_result.get('severity_level')}; "
                f"Gap(SVM)={score_signals.get('Gap_SVM')}; "
                f"μ(predicted)={score_signals.get('Mu_predicted_FCM')}; "
                f"ARIMA={score_signals.get('ARIMA_trend')}; "
                f"SHAP={score_signals.get('SHAP_reliability')}; "
                f"Total={total_score}; action_required={action_required}"
            ),
        )

        # ==============================
        # Step 12.1: Mandatory Explicit RAG Majority Vote
        # ==============================

        rag_decision_result = {
            "success": False,
            "rag_used": False,
            "case_type": "not_evaluated",
            "recommended_route": "not_evaluated",
            "reason": "RAG majority vote has not been evaluated yet.",
        }

        with ls_trace(
            "Step 12.1 - Mandatory Explicit RAG Majority Vote",
            "chain",
            inputs={
                "total_action_score": total_score,
                "svm_confidence": extracted_confidence,
                "shap_cosine_similarity": shap_cosine_similarity,
                "forecast_risk_score": forecast_risk_score,
                "expected_change_pct": expected_change_pct,
                "rag_query_prepared": bool(rag_query),
            },
        ) as trace:
            try:
                borderline_lower = 0.55
                borderline_upper = 0.75
                is_borderline_case = borderline_lower <= float(total_score) < borderline_upper

                if is_borderline_case:
                    # Retrieval is executed only for borderline scores, as required.
                    rag_result = retrieve_rag_context(
                        rag_query,
                        top_k=3,
                    )

                    if isinstance(rag_result, dict):
                        rag_result["retrieval_executed"] = True
                        rag_result["retrieval_policy"] = "Executed because the case is borderline."

                else:
                    if isinstance(rag_result, dict):
                        rag_result["retrieval_executed"] = False
                        rag_result["retrieval_policy"] = (
                            "Not executed because the score is outside the "
                            "borderline range; the mandatory RAG decision layer "
                            "still logged the not_borderline result."
                        )

                rag_decision_result = run_explicit_rag_majority_vote(
                    rag_result=rag_result,
                    total_action_score=total_score,
                    svm_confidence=extracted_confidence,
                    shap_cosine_similarity=shap_cosine_similarity,
                    forecast_risk_score=forecast_risk_score,
                    expected_change_pct=expected_change_pct,
                    lower_borderline=borderline_lower,
                    upper_borderline=borderline_upper,
                )

            except Exception as e:
                rag_decision_result = {
                    "success": False,
                    "rag_used": False,
                    "case_type": "rag_decision_error",
                    "recommended_route": "human_review",
                    "majority_vote": "error",
                    "tie_detected": False,
                    "zero_success": True,
                    "reason": (
                        "Explicit RAG majority vote failed, so the case is routed "
                        "to human review for safety."
                    ),
                    "error": str(e),
                }

            # Minimal connection to the current action layer until the full ReAct
            # routing engine is added in the next phase.
            pre_rag_action_required = bool(action_required)
            rag_recommended_route = rag_decision_result.get("recommended_route")

            if rag_decision_result.get("rag_used"):
                if rag_recommended_route == "dispatch":
                    action_required = True
                elif rag_recommended_route in {"human_review", "shap_recheck", "monitor"}:
                    action_required = False

            rag_decision_result["pre_rag_action_required"] = pre_rag_action_required
            rag_decision_result["post_rag_action_required"] = bool(action_required)
            rag_decision_result["action_required_adjusted_by_rag"] = (
                pre_rag_action_required != bool(action_required)
            )

            end_trace(
                trace,
                {
                    "rag_used": rag_decision_result.get("rag_used"),
                    "case_type": rag_decision_result.get("case_type"),
                    "retrieved_count": len(rag_result.get("retrieved_items", []))
                    if isinstance(rag_result, dict)
                    else 0,
                    "majority_vote": rag_decision_result.get("majority_vote"),
                    "vote_counts": rag_decision_result.get("vote_counts"),
                    "tie_detected": rag_decision_result.get("tie_detected"),
                    "zero_success": rag_decision_result.get("zero_success"),
                    "arima_alignment": rag_decision_result.get("arima_alignment"),
                    "recommended_route": rag_decision_result.get("recommended_route"),
                    "pre_rag_action_required": pre_rag_action_required,
                    "post_rag_action_required": action_required,
                    "error": rag_decision_result.get("error"),
                },
            )

        step(
            "Mandatory Explicit RAG Majority Vote",
            "success" if rag_decision_result.get("success") else "warning",
            (
                f"case_type={rag_decision_result.get('case_type')}; "
                f"rag_used={rag_decision_result.get('rag_used')}; "
                f"majority_vote={rag_decision_result.get('majority_vote')}; "
                f"route={rag_decision_result.get('recommended_route')}; "
                f"tie={rag_decision_result.get('tie_detected')}; "
                f"zero_success={rag_decision_result.get('zero_success')}; "
                f"post_rag_action_required={action_required}"
            ),
        )

        # ==============================
        # Step 12.2: Formal ReAct Routing Engine
        # ==============================

        react_routing_result = {
            "success": False,
            "route_id": "REACT_NOT_EVALUATED",
            "route_name": "ReAct Not Evaluated",
            "recommended_route": "human_review",
            "action_required": False,
            "reason": "ReAct routing has not been evaluated yet.",
            "adaptive_action": {},
            "react_trace": {},
            "routes_evaluated": [],
        }

        with ls_trace(
            "Step 12.2 - Formal ReAct Routing Engine",
            "chain",
            inputs={
                "total_action_score": total_score,
                "pre_react_action_required": action_required,
                "svm_confidence": extracted_confidence,
                "dynamic_severity": dynamic_severity_score,
                "shap_cosine_similarity": shap_cosine_similarity,
                "shap_reliability_level": shap_reliability_result.get("reliability_level"),
                "rag_case_type": rag_decision_result.get("case_type"),
                "rag_used": rag_decision_result.get("rag_used"),
                "rag_majority_vote": rag_decision_result.get("majority_vote"),
                "rag_recommended_route": rag_decision_result.get("recommended_route"),
                "react_route": react_routing_result.get("route_id"),
                "react_route_name": react_routing_result.get("route_name"),
                "react_recommended_route": react_routing_result.get("recommended_route"),
                "adaptive_action": react_routing_result.get("adaptive_action"),
                "forecast_risk_score": forecast_risk_score,
                "expected_change_pct": expected_change_pct,
                "dominant_shap_feature": dominant_shap_feature,
            },
        ) as trace:
            try:
                pre_react_action_required = bool(action_required)

                react_routing_result = run_react_routing_engine(
                    total_action_score=total_score,
                    action_required=action_required,
                    svm_confidence=extracted_confidence,
                    dynamic_severity=dynamic_severity_score,
                    shap_cosine_similarity=shap_cosine_similarity,
                    shap_reliability_level=str(shap_reliability_result.get("reliability_level", "unknown")),
                    rag_majority_result=rag_decision_result,
                    forecast_risk_score=forecast_risk_score,
                    expected_change_pct=expected_change_pct,
                    dominant_shap_feature=dominant_shap_feature,
                    predicted_class=extracted_prediction,
                    decoded_plain_text=decoded_plain_text,
                )

                # ReAct is now the final action gate.
                action_required = bool(react_routing_result.get("action_required", False))

            except Exception as e:
                pre_react_action_required = bool(action_required)
                action_required = False

                react_routing_result = {
                    "success": False,
                    "method": "formal_react_routing_engine",
                    "route_id": "REACT_ERROR",
                    "route_name": "ReAct Routing Error",
                    "recommended_route": "human_review",
                    "action_required": False,
                    "reason": "ReAct routing failed; routed to human review for safety.",
                    "adaptive_action": {
                        "action_type": "human_review",
                        "timing": "immediate_manual_review",
                        "intensity": "high",
                        "personalized_message": (
                            "ReAct routing failed during inference. Route this case "
                            "to human review and inspect pipeline logs."
                        ),
                    },
                    "react_trace": {
                        "observe": {},
                        "reason": {"error": str(e)},
                        "act": {"recommended_route": "human_review"},
                        "observe_after_action": {"status": "error_logged"},
                    },
                    "routes_evaluated": [],
                    "error": str(e),
                }

            react_routing_result["pre_react_action_required"] = bool(pre_react_action_required)
            react_routing_result["post_react_action_required"] = bool(action_required)
            react_routing_result["action_required_adjusted_by_react"] = (
                bool(pre_react_action_required) != bool(action_required)
            )

            end_trace(
                trace,
                {
                    "route_id": react_routing_result.get("route_id"),
                    "route_name": react_routing_result.get("route_name"),
                    "recommended_route": react_routing_result.get("recommended_route"),
                    "reason": react_routing_result.get("reason"),
                    "pre_react_action_required": pre_react_action_required,
                    "post_react_action_required": action_required,
                    "adaptive_action": react_routing_result.get("adaptive_action"),
                    "routes_evaluated": react_routing_result.get("routes_evaluated"),
                    "error": react_routing_result.get("error"),
                },
            )

        adaptive_action = react_routing_result.get("adaptive_action", {})

        step(
            "Formal ReAct Routing Engine",
            "success" if react_routing_result.get("success") else "warning",
            (
                f"route={react_routing_result.get('route_id')} "
                f"({react_routing_result.get('route_name')}); "
                f"recommended={react_routing_result.get('recommended_route')}; "
                f"action_required={action_required}; "
                f"intensity={adaptive_action.get('intensity')}; "
                f"timing={adaptive_action.get('timing')}"
            ),
        )

        # ==============================
        # Step 13: SHAP English Explanation Node
        # ==============================

        with ls_trace(
            "Step 13 - SHAP English Explanation Node",
            "llm",
            inputs={
                "prediction": extracted_prediction,
                "confidence": extracted_confidence,
                "shap_score": shap_score,
                "shap_cosine_similarity": shap_cosine_similarity,
                "shap_reliability_level": shap_reliability_result.get("reliability_level"),
                "xai_score": xai_score,
            },
        ) as trace:
            shap_english_result = generate_shap_english_explanation(
                predicted_cipher_type=extracted_prediction,
                svm_confidence=extracted_confidence,
                shap_score=shap_score,
                xai_score=xai_score,
                importance_df=importance_df,
                cypher_text=extracted_cypher_text,
                decoded_plain_text=decoded_plain_text,
                use_llm=True,
            )

            end_trace(
                trace,
                {
                    "engine": shap_english_result.get("engine"),
                    "explanation_preview": shap_english_result.get("explanation", "")[:1000],
                    "top_features": shap_english_result.get("top_features", []),
                    "llm_error": shap_english_result.get("llm_error"),
                },
            )

        step(
            "SHAP English Explanation Node",
            "success",
            f"Generated English SHAP explanation using engine={shap_english_result.get('engine')}.",
        )

        # ==============================
        # Step 14: LLM/Fallback Report
        # ==============================

        with ls_trace(
            "Step 14 - LLM/Fallback Report Generator",
            "llm",
            inputs={
                "avg_svm_confidence": avg_svm_confidence,
                "shap_score": shap_score,
                "shap_recheck_result": shap_recheck_result,
                "xai_score": xai_score,
                "forecast_risk_score": forecast_risk_score,
                "drift_result": drift_result,
                "total_action_score": total_score,
                "action_required": action_required,
                "dynamic_severity": dynamic_severity_score,
                "dynamic_severity_level": dynamic_severity_result.get("severity_level"),
                "rag_decision_result": rag_decision_result,
                "react_routing_result": react_routing_result,
                "final_react_route": react_routing_result.get("route_id"),
                "adaptive_action": react_routing_result.get("adaptive_action"),
                "decoded_plain_text": decoded_plain_text,
            },
        ) as trace:
            if importance_df is not None and not importance_df.empty:
                top_reasons = importance_df.head(5).to_string(index=False)
            else:
                top_reasons = "SHAP not available."

            report = generate_crypto_ai_report(
                avg_svm_confidence=avg_svm_confidence,
                shap_score=shap_score,
                top_shap_reasons=top_reasons,
                forecast_risk_score=forecast_risk_score,
                expected_change_pct=sarima_result.get("expected_change_pct", 0.0),
                drift_result=drift_result,
                total_action_score=total_score,
                action_required=action_required,
                decoded_agent_command=decoded_plain_text or extracted_cypher_text,
                encryption_type=f"SVM predicted: {extracted_prediction}",
                cipher_reason=(
                    f"Extractor: {extraction_result.get('reason', '')}. "
                    f"SHAP English Explanation: {shap_english_result.get('explanation', '')}. "
                    f"SHAP Re-check: triggered={shap_recheck_result.get('recheck_triggered')}, "
                    f"final_cosine={shap_recheck_result.get('final_shap_cosine_similarity')}, "
                    f"route_after_recheck={shap_recheck_result.get('route_after_recheck')}. "
                    f"Dynamic Severity: {dynamic_severity_score} "
                    f"({dynamic_severity_result.get('severity_level')}). "
                    f"Severity reasons: {'; '.join(dynamic_severity_result.get('reason_codes', [])[:4])}. "
                    f"Explicit RAG decision: case_type={rag_decision_result.get('case_type')}, "
                    f"rag_used={rag_decision_result.get('rag_used')}, "
                    f"majority_vote={rag_decision_result.get('majority_vote')}, "
                    f"route={rag_decision_result.get('recommended_route')}, "
                    f"reason={rag_decision_result.get('reason')}. "
                    f"Final ReAct route: {react_routing_result.get('route_id')} "
                    f"({react_routing_result.get('route_name')}); "
                    f"adaptive action={react_routing_result.get('adaptive_action', {}).get('action_type')}; "
                    f"timing={react_routing_result.get('adaptive_action', {}).get('timing')}; "
                    f"intensity={react_routing_result.get('adaptive_action', {}).get('intensity')}. "
                    f"RAG context: {rag_result.get('retrieved_context', '')}"
                ),
            )

            end_trace(
                trace,
                {
                    "report_length": len(str(report)),
                    "report_preview": str(report)[:1200],
                },
            )

        step(
            "LLM/Fallback Report",
            "success",
            "Generated final report for dashboard and API action.",
        )

        # ==============================
        # Step 15: API Action
        # ==============================

        api_result = {
            "success": False,
            "message": "API action not executed because score is below threshold.",
            "email_attempted": False,
            "email_sent": False,
            "action_logged": False,
        }

        with ls_trace(
            "Step 15 - API Action",
            "tool",
            inputs={
                "execute_api_action": execute_api_action,
                "force_api_action": force_api_action,
                "action_required": action_required,
                "total_action_score": total_score,
                "rag_decision_route": rag_decision_result.get("recommended_route"),
                "rag_used": rag_decision_result.get("rag_used"),
                "react_route": react_routing_result.get("route_id"),
                "react_recommended_route": react_routing_result.get("recommended_route"),
                "adaptive_action": react_routing_result.get("adaptive_action"),
                "shap_recheck_triggered": shap_recheck_result.get("recheck_triggered"),
                "shap_recheck_route": shap_recheck_result.get("route_after_recheck"),
            },
        ) as trace:
            if execute_api_action and (action_required or force_api_action):
                api_result = send_email_alert(
                    "CRYPTO AI Cipher Intelligence Alert",
                    report,
                )

                step(
                    "API Action",
                    "success" if api_result.get("success") else "warning",
                    api_result.get("message", ""),
                )

                end_trace(
                    trace,
                    {
                        "api_executed": True,
                        "api_result": api_result,
                    },
                )
            else:
                step(
                    "API Action",
                    "info",
                    api_result["message"],
                )

                end_trace(
                    trace,
                    {
                        "api_executed": False,
                        "api_result": api_result,
                    },
                )

        # ==============================
        # Step 15.1: Outcome Tracking + Between-Cycle Adaptation
        # ==============================

        outcome_tracking_result = {
            "success": False,
            "method": "outcome_tracking_not_executed",
            "outcome_status": "not_tracked",
            "logged": False,
            "adaptation": {},
            "observe_after_action": {},
        }

        with ls_trace(
            "Step 15.1 - Outcome Tracking and Between-Cycle Adaptation",
            "chain",
            inputs={
                "prediction": extracted_prediction,
                "svm_confidence": extracted_confidence,
                "total_action_score": total_score,
                "dynamic_severity": dynamic_severity_score,
                "shap_cosine_similarity": shap_cosine_similarity,
                "react_route": react_routing_result.get("route_id"),
                "react_recommended_route": react_routing_result.get("recommended_route"),
                "api_success": bool(api_result.get("success")),
                "outcome_status": outcome_tracking_result.get("outcome_status"),
                "outcome_logged": outcome_tracking_result.get("logged"),
                "outcome_adaptation": outcome_tracking_result.get("adaptation"),
            },
        ) as trace:
            try:
                outcome_tracking_result = log_and_adapt_outcome(
                    cypher_text=extracted_cypher_text,
                    decoded_plain_text=decoded_plain_text,
                    predicted_class=extracted_prediction,
                    svm_confidence=extracted_confidence,
                    total_action_score=total_score,
                    dynamic_severity=dynamic_severity_score,
                    shap_cosine_similarity=shap_cosine_similarity,
                    rag_decision_result=rag_decision_result,
                    react_routing_result=react_routing_result,
                    api_result=api_result,
                    action_required=action_required,
                    feedback="",
                )

                # Attach the post-action observation to the existing ReAct trace.
                try:
                    react_trace = react_routing_result.setdefault("react_trace", {})
                    react_trace["observe_after_action"] = outcome_tracking_result.get(
                        "observe_after_action",
                        {},
                    )
                    react_routing_result["outcome_tracking_result"] = outcome_tracking_result
                except Exception:
                    pass

            except Exception as e:
                outcome_tracking_result = {
                    "success": False,
                    "method": "outcome_tracking_and_between_cycle_adaptation",
                    "outcome_status": "tracking_error",
                    "logged": False,
                    "error": str(e),
                    "adaptation": {
                        "recommendation": "tracking_error",
                        "reason_codes": [
                            "Outcome tracking failed; inspect pipeline logs."
                        ],
                    },
                    "observe_after_action": {
                        "status": "tracking_error",
                        "error": str(e),
                    },
                }

            end_trace(
                trace,
                {
                    "case_id": outcome_tracking_result.get("case_id"),
                    "outcome_status": outcome_tracking_result.get("outcome_status"),
                    "logged": outcome_tracking_result.get("logged"),
                    "log_path": outcome_tracking_result.get("log_path"),
                    "adaptation": outcome_tracking_result.get("adaptation"),
                    "history_count_after_logging": outcome_tracking_result.get(
                        "history_count_after_logging"
                    ),
                    "error": outcome_tracking_result.get("error")
                    or outcome_tracking_result.get("log_error"),
                },
            )

        adaptation_payload = outcome_tracking_result.get("adaptation", {}) or {}

        step(
            "Outcome Tracking + Between-Cycle Adaptation",
            "success" if outcome_tracking_result.get("success") else "warning",
            (
                f"outcome={outcome_tracking_result.get('outcome_status')}; "
                f"logged={outcome_tracking_result.get('logged')}; "
                f"adaptation={adaptation_payload.get('recommendation')}; "
                f"bias={adaptation_payload.get('action_bias')}"
            ),
        )

        # ==============================
        # Build Display DataFrames
        # ==============================

        display_live_df = live_df.copy()

        for col in training_labeled_df.columns:
            if col not in display_live_df.columns:
                display_live_df[col] = np.nan

        display_live_df = display_live_df[training_labeled_df.columns]

        labeled_df = pd.concat(
            [
                training_labeled_df,
                display_live_df,
            ],
            ignore_index=True,
        )

        expanded_df = training_labeled_df.copy()
        cleaned_df = training_labeled_df.copy()

        svm_metrics = {
            "accuracy": metadata.get("svm_accuracy", 0.0),
            "classes": metadata.get("svm_classes", []),
            "mode": "loaded_saved_model_live_inference",
        }

        result = {
            "live_mode": True,
            "training_mode": False,
            "svm_retrained_this_run": False,
            "artifact_status": artifact_status,
            "artifacts_metadata": metadata,
            "timeline": timeline,
            "extraction_result": extraction_result,
            "raw_df": raw_df,
            "working_df": working_df,
            "expanded_df": expanded_df,
            "cleaned_df": cleaned_df,
            "labeled_df": labeled_df,
            "training_labeled_df": training_labeled_df,
            "live_df": live_df,
            "feature_cols": feature_cols,
            "risk_feature": risk_feature,
            "date_col": date_col,
            "forecast_target": forecast_target,
            "X_scaled_shape": live_scaled.shape,
            "scaler": scaler,
            "dbscan_model": dbscan_model,
            "fcm_info": fcm_info,
            "svm_model": svm_model,
            "svm_metrics": svm_metrics,
            "extracted_prediction": extracted_prediction,
            "svm_original_prediction": svm_original_prediction,
            "cipher_validation_result": cipher_validation_result,
            "extracted_confidence": extracted_confidence,
            "decode_result": decode_result,
            "decoded_plain_text": decoded_plain_text,
            "rag_result": rag_result,
            "rag_decision_result": rag_decision_result,
            "react_routing_result": react_routing_result,
            "final_react_route": react_routing_result.get("route_id"),
            "final_react_route_name": react_routing_result.get("route_name"),
            "adaptive_action": react_routing_result.get("adaptive_action", {}),
            "shap_result": shap_result,
            "shap_reliability_result": shap_reliability_result,
            "shap_recheck_result": shap_recheck_result,
            "shap_cosine_similarity": shap_cosine_similarity,
            "shap_english_result": shap_english_result,
            "sarima_result": sarima_result,
            "drift_result": drift_result,
            "dual_drift_result": dual_drift_result,
            "dynamic_reclustering_result": dynamic_reclustering_result,
            "outer_react_loop_result": outer_react_loop_result,
            "outer_loop_triggered": outer_react_loop_result.get("outer_loop_triggered"),
            "outer_loop_decision": outer_react_loop_result.get("decision"),
            "outer_loop_accepted_candidate": outer_react_loop_result.get("accepted_candidate"),
            "rollback_required": outer_react_loop_result.get("rollback_required"),
            "avg_svm_confidence": avg_svm_confidence,
            "action_score_result": action_score_result,
            "dynamic_severity_result": dynamic_severity_result,
            "dynamic_severity_score": dynamic_severity_score,
            "dynamic_severity_level": dynamic_severity_result.get("severity_level"),
            "xai_score": xai_score,
            "total_action_score": total_score,
            "action_required": action_required,
            "api_result": api_result,
            "outcome_tracking_result": outcome_tracking_result,
            "outcome_status": outcome_tracking_result.get("outcome_status"),
            "outcome_adaptation": outcome_tracking_result.get("adaptation", {}),
            "report": report,
        }

        end_trace(
            root_run,
            {
                "final_status": "completed",
                "live_mode": True,
                "svm_retrained_this_run": False,
                "extracted_cypher_text": extracted_cypher_text,
                "svm_predicted_cipher_type": extracted_prediction,
                "svm_original_prediction": svm_original_prediction,
                "cipher_validation_override": cipher_validation_result.get("override_applied"),
                "cipher_validation_confidence": cipher_validation_result.get("deterministic_confidence"),
                "extracted_confidence": extracted_confidence,
                "decoded_plain_text": decoded_plain_text,
                "shap_score": shap_score,
                "xai_score": xai_score,
                "shap_english_engine": shap_english_result.get("engine"),
                "shap_cosine_similarity": shap_cosine_similarity,
                "shap_reliability_level": shap_reliability_result.get("reliability_level"),
                "shap_recheck_triggered": shap_recheck_result.get("recheck_triggered"),
                "shap_recheck_route": shap_recheck_result.get("route_after_recheck"),
                "dynamic_severity": dynamic_severity_score,
                "dynamic_severity_level": dynamic_severity_result.get("severity_level"),
                "rag_case_type": rag_decision_result.get("case_type"),
                "rag_used": rag_decision_result.get("rag_used"),
                "rag_majority_vote": rag_decision_result.get("majority_vote"),
                "rag_recommended_route": rag_decision_result.get("recommended_route"),
                "react_route": react_routing_result.get("route_id"),
                "react_route_name": react_routing_result.get("route_name"),
                "react_recommended_route": react_routing_result.get("recommended_route"),
                "adaptive_action": react_routing_result.get("adaptive_action"),
                "forecast_risk_score": forecast_risk_score,
                "drift_detected": bool(drift_result.get("drift_detected")),
                "dual_drift_detected": bool(dual_drift_result.get("dual_drift_detected")),
                "dual_drift_decision": dual_drift_result.get("decision"),
                "outer_loop_required": bool(dual_drift_result.get("outer_loop_required")),
                "total_action_score": total_score,
                "action_required": action_required,
                "api_success": bool(api_result.get("success")),
                "outcome_status": outcome_tracking_result.get("outcome_status"),
                "outcome_logged": outcome_tracking_result.get("logged"),
                "outcome_adaptation": outcome_tracking_result.get("adaptation"),
            },
        )

        return result