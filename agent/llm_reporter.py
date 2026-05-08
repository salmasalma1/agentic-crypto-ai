"""Optional LLM report generator with deterministic fallback."""

import os
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


def fallback_report(**kwargs) -> str:
    top_shap_reasons = kwargs.get("top_shap_reasons", "Not available")
    drift_result: Dict[str, Any] = kwargs.get("drift_result", {}) or {}
    return f"""
CRYPTO AI - Automated Cipher Intelligence Alert

Input / Decoded Command:
{kwargs.get('decoded_agent_command', 'No command provided')}

Predicted Cipher / Encoding:
{kwargs.get('encryption_type', 'Not available')}

Classification Evidence:
- Average SVM confidence: {float(kwargs.get('avg_svm_confidence', 0.0)):.2f}
- SHAP importance score: {float(kwargs.get('shap_score', 0.0)):.2f}
- Forecast risk score: {float(kwargs.get('forecast_risk_score', 0.0)):.2f}
- Expected change percentage: {kwargs.get('expected_change_pct', 0.0)}

Top SHAP Reasons:
{top_shap_reasons}

Drift Detection:
- Drift detected: {drift_result.get('drift_detected', False)}
- Drift score: {drift_result.get('drift_score', 0.0)}
- Recommendation: {drift_result.get('recommendation', '')}

Total Action Score:
{float(kwargs.get('total_action_score', 0.0)):.2f}

Final Decision:
{"Action is required. Send the alert and investigate this encoded message." if kwargs.get('action_required', False) else "No immediate action is required."}
""".strip()


def _get_llm_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not endpoint or not api_key or not deployment:
        return None
    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore
        from pydantic import SecretStr
        return AzureChatOpenAI(azure_endpoint=endpoint, api_key=SecretStr(api_key), azure_deployment=deployment, api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"), temperature=0.2)
    except Exception:
        return None


def generate_crypto_ai_report(**kwargs) -> str:
    llm = _get_llm_client()
    if llm is None:
        return fallback_report(**kwargs)
    prompt = f"""
You are the reporting layer of CRYPTO AI. Write a concise professional alert report.
Use the following model outputs; do not claim that the LLM classified or decoded the cipher.
{fallback_report(**kwargs)}
"""
    try:
        return str(llm.invoke(prompt).content)
    except Exception:
        return fallback_report(**kwargs)
