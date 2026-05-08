# pipeline/shap_english_explainer.py

"""
shap_english_explainer.py

This file adds a new AI/Rule-based explanation node after SHAP.

Goal:
- Take SHAP outputs.
- Convert the technical feature importance into a clear English explanation.
- Display the explanation in Streamlit.
- Use LLM if configured.
- Use deterministic fallback if LLM credentials are missing.

Important:
This node does NOT replace SHAP.
It only explains SHAP results in English.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


# ==============================
# LLM Client
# ==============================

def get_llm_client():
    """
    Creates Azure OpenAI client if credentials exist.

    Required .env:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT

    Optional:
    AZURE_OPENAI_API_VERSION
    """

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or "gpt-4o"
    api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2024-05-01-preview"
    ).strip()

    if not endpoint or not api_key or not deployment:
        return None

    try:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=SecretStr(api_key),
            azure_deployment=deployment,
            api_version=api_version,
            temperature=0.2,
        )

    except Exception:
        return None


# ==============================
# Feature Interpretation Helpers
# ==============================

def humanize_feature_name(feature_name: str) -> str:
    """
    Converts technical feature names into readable English labels.
    """

    feature_name = str(feature_name or "").strip()

    mapping = {
        "cypher_length": "encoded text length",
        "cypher_entropy": "character randomness / entropy",
        "cypher_digit_count": "number of digits",
        "cypher_alpha_count": "number of alphabetic characters",
        "cypher_upper_count": "uppercase character count",
        "cypher_lower_count": "lowercase character count",
        "cypher_symbol_count": "symbol count",
        "cypher_space_count": "space count",
        "cypher_equal_count": "padding equals signs",
        "cypher_plus_slash_count": "Base64 plus/slash symbols",
        "cypher_dot_dash_count": "Morse dot-dash symbols",
        "cypher_digit_ratio": "digit ratio",
        "cypher_alpha_ratio": "alphabetic character ratio",
        "cypher_upper_ratio": "uppercase character ratio",
        "cypher_lower_ratio": "lowercase character ratio",
        "cypher_symbol_ratio": "symbol ratio",
        "cypher_space_ratio": "space ratio",
        "cypher_base64_ratio": "Base64 character-pattern ratio",
        "cypher_base32_ratio": "Base32 character-pattern ratio",
        "cypher_morse_ratio": "Morse character-pattern ratio",
        "cypher_dot_dash_ratio": "dot-dash ratio",
    }

    if feature_name in mapping:
        return mapping[feature_name]

    return feature_name.replace("_", " ")


def get_top_shap_features(
    importance_df: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extracts top SHAP feature importances safely.
    """

    if importance_df is None or importance_df.empty:
        return []

    required_cols = {"feature", "shap_importance"}

    if not required_cols.issubset(set(importance_df.columns)):
        return []

    top_df = (
        importance_df
        .sort_values("shap_importance", ascending=False)
        .head(top_n)
        .copy()
    )

    top_features = []

    for _, row in top_df.iterrows():
        feature = str(row.get("feature", ""))
        importance = float(row.get("shap_importance", 0.0))

        top_features.append({
            "feature": feature,
            "readable_feature": humanize_feature_name(feature),
            "shap_importance": round(importance, 6),
        })

    return top_features


def infer_reason_sentence(predicted_cipher_type: str, top_features: List[Dict[str, Any]]) -> str:
    """
    Builds a deterministic explanation sentence based on prediction and top SHAP features.
    """

    predicted = str(predicted_cipher_type or "Unknown").strip()

    feature_names = [
        item["feature"]
        for item in top_features
    ]

    if predicted.lower() == "morse":
        if any("morse" in feature or "dot_dash" in feature for feature in feature_names):
            return (
                "The decision is mainly supported by dot-dash and Morse-pattern features, "
                "which indicate that the input contains symbols commonly used in Morse code."
            )

        return (
            "The model predicted Morse based on the overall encoded-text structure "
            "and the strongest SHAP-ranked character features."
        )

    if predicted.lower() == "base32":
        if any("base32" in feature or "upper" in feature or "equal" in feature for feature in feature_names):
            return (
                "The decision is mainly supported by Base32-style character patterns, "
                "uppercase characters, and possible padding symbols."
            )

        return (
            "The model predicted Base32 because the strongest SHAP-ranked features "
            "match the structure of Base32 encoded text."
        )

    if predicted.lower() == "base64":
        if any("base64" in feature or "lower" in feature or "plus_slash" in feature or "equal" in feature for feature in feature_names):
            return (
                "The decision is mainly supported by Base64-style character patterns, "
                "including mixed alphanumeric structure and possible padding."
            )

        return (
            "The model predicted Base64 because the strongest SHAP-ranked features "
            "match the structure of Base64 encoded text."
        )

    if predicted.lower() == "anomaly":
        return (
            "The model treated this input as unusual because its feature pattern does not "
            "strongly match the normal known cipher families."
        )

    return (
        "The model prediction is explained by the highest-ranked SHAP features, "
        "which had the strongest influence on the SVM decision."
    )


# ==============================
# Rule-Based Fallback Explanation
# ==============================

def build_rule_based_shap_explanation(
    predicted_cipher_type: str,
    svm_confidence: float,
    shap_score: float,
    xai_score: float,
    top_features: List[Dict[str, Any]],
    decoded_plain_text: str = "",
) -> str:
    """
    Builds a clear English SHAP explanation without LLM.
    """

    predicted_cipher_type = str(predicted_cipher_type or "Unknown").strip()

    try:
        svm_confidence = float(svm_confidence)
    except Exception:
        svm_confidence = 0.0

    try:
        shap_score = float(shap_score)
    except Exception:
        shap_score = 0.0

    try:
        xai_score = float(xai_score)
    except Exception:
        xai_score = 0.0

    if svm_confidence >= 0.85:
        confidence_word = "high"
    elif svm_confidence >= 0.60:
        confidence_word = "moderate"
    else:
        confidence_word = "low"

    if shap_score >= 0.50:
        importance_word = "strong"
    elif shap_score >= 0.25:
        importance_word = "moderate"
    else:
        importance_word = "limited"

    top_feature_lines = []

    for idx, item in enumerate(top_features, start=1):
        top_feature_lines.append(
            f"{idx}. {item['readable_feature']} "
            f"(SHAP importance: {item['shap_importance']})"
        )

    if not top_feature_lines:
        top_feature_text = "No detailed SHAP feature ranking was available."
    else:
        top_feature_text = "\n".join(top_feature_lines)

    reason_sentence = infer_reason_sentence(
        predicted_cipher_type=predicted_cipher_type,
        top_features=top_features,
    )

    decoded_section = ""
    if decoded_plain_text:
        decoded_section = (
            f"\n\nDecoded output:\n"
            f"The deterministic decoder produced: \"{decoded_plain_text}\"."
        )

    explanation = f"""
SHAP English Explanation

The SVM-RBF classifier predicted this live input as: {predicted_cipher_type}.

The prediction confidence is {svm_confidence:.4f}, which indicates a {confidence_word} level of certainty.
The SHAP importance score is {shap_score:.4f}, showing a {importance_word} concentration of explanation strength.
The XAI score is calculated as:

XAI Score = Confidence × Importance
XAI Score = {svm_confidence:.4f} × {shap_score:.4f} = {xai_score:.4f}

Main explanation:
{reason_sentence}

Top SHAP features influencing the decision:
{top_feature_text}
{decoded_section}

Interpretation:
These features show which numerical characteristics of the encoded text had the strongest influence on the SVM decision. The LLM is not making the classification here; it is only converting SHAP outputs into a readable English explanation.
""".strip()

    return explanation


# ==============================
# LLM Explanation
# ==============================

def build_llm_shap_explanation(
    predicted_cipher_type: str,
    svm_confidence: float,
    shap_score: float,
    xai_score: float,
    top_features: List[Dict[str, Any]],
    cypher_text: str = "",
    decoded_plain_text: str = "",
) -> str:
    """
    Uses LLM to explain SHAP outputs in clear English.

    If the LLM is unavailable or fails, the function caller should use fallback.
    """

    llm = get_llm_client()

    if llm is None:
        raise RuntimeError("LLM client is not configured.")

    prompt = f"""
You are the SHAP English Explanation Node inside CRYPTO AI.

Your task:
Explain the SVM-RBF model decision in clear English using SHAP outputs.

Important rules:
- Do not change the model prediction.
- Do not claim that the LLM classified the cipher.
- The SVM-RBF classifier made the prediction.
- SHAP provided the explanation.
- Keep the explanation clear, professional, and suitable for a Streamlit dashboard.
- Mention the XAI formula: XAI Score = Confidence × Importance.
- Do not invent features not listed below.

Model output:
- Predicted cipher/state: {predicted_cipher_type}
- SVM confidence: {svm_confidence}
- SHAP importance score: {shap_score}
- XAI score: {xai_score}
- Cypher text: {cypher_text}
- Decoded plain text: {decoded_plain_text}

Top SHAP features:
{top_features}

Write the explanation using this structure:
1. Final prediction
2. Confidence and XAI score
3. Main SHAP evidence
4. Plain-English interpretation
5. One sentence confirming that SVM performed the classification and the LLM only explains the SHAP result
"""

    response = llm.invoke(prompt)

    return str(response.content).strip()


# ==============================
# Main Node Function
# ==============================

def generate_shap_english_explanation(
    predicted_cipher_type: str,
    svm_confidence: float,
    shap_score: float,
    xai_score: float,
    importance_df: pd.DataFrame,
    cypher_text: str = "",
    decoded_plain_text: str = "",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Main SHAP English Explanation Node.

    Inputs:
    - predicted_cipher_type
    - svm_confidence
    - shap_score
    - xai_score
    - importance_df
    - cypher_text
    - decoded_plain_text
    - use_llm

    Output:
    {
        "explanation": "...",
        "engine": "llm" | "rule_based_fallback",
        "top_features": [...],
        "success": True/False
    }
    """

    top_features = get_top_shap_features(
        importance_df=importance_df,
        top_n=5,
    )

    if use_llm:
        try:
            explanation = build_llm_shap_explanation(
                predicted_cipher_type=predicted_cipher_type,
                svm_confidence=svm_confidence,
                shap_score=shap_score,
                xai_score=xai_score,
                top_features=top_features,
                cypher_text=cypher_text,
                decoded_plain_text=decoded_plain_text,
            )

            return {
                "success": True,
                "engine": "llm",
                "explanation": explanation,
                "top_features": top_features,
            }

        except Exception as e:
            fallback_explanation = build_rule_based_shap_explanation(
                predicted_cipher_type=predicted_cipher_type,
                svm_confidence=svm_confidence,
                shap_score=shap_score,
                xai_score=xai_score,
                top_features=top_features,
                decoded_plain_text=decoded_plain_text,
            )

            return {
                "success": True,
                "engine": "rule_based_fallback",
                "explanation": fallback_explanation,
                "top_features": top_features,
                "llm_error": str(e),
            }

    fallback_explanation = build_rule_based_shap_explanation(
        predicted_cipher_type=predicted_cipher_type,
        svm_confidence=svm_confidence,
        shap_score=shap_score,
        xai_score=xai_score,
        top_features=top_features,
        decoded_plain_text=decoded_plain_text,
    )

    return {
        "success": True,
        "engine": "rule_based_fallback",
        "explanation": fallback_explanation,
        "top_features": top_features,
    }